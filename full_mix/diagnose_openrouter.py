#!/usr/bin/env python3
"""Stress / diagnose the OpenRouter judge endpoint (deepseek-v4-flash).

Sends N chat-completions with the EXACT payload shape that
`full_mix/common/judge_client.call_judge` uses (same response_format,
same enable_thinking extra_body), at a configurable concurrency, and
reports a breakdown of *why* requests fail — distinguishing the OpenRouter
failure modes that a naive "did it raise?" test misses:

  * 200 OK but choices=None/[]  -> upstream provider rate-limit/refusal
                                   (real reason is in resp.error)
  * 200 OK but empty content    -> truncated / content-filtered
                                   (finish_reason tells you which)
  * timeouts / connection errors
  * 429 / 5xx  HTTP status
  * JSONDecodeError              -> proxy returned a malformed body
  * response_format unsupported  -> provider rejects json_object

Config comes from the same env vars as training. If they aren't already
exported, the FULL_MIX_JUDGE_* ones are auto-loaded from runtime_env.yaml.

Examples
--------
  # 100 requests, 16 in flight (matches FULL_MIX_JUDGE_CONCURRENCY)
  python full_mix/diagnose_openrouter.py -n 100 -c 16

  # isolate whether response_format/json_object is the culprit
  python full_mix/diagnose_openrouter.py -n 50 --no-json
  python full_mix/diagnose_openrouter.py -n 50 --no-thinking

  # pull OpenRouter's per-generation stats (provider, native finish reason)
  python full_mix/diagnose_openrouter.py -n 30 --gen-stats
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_ENV = os.path.join(_HERE, "runtime_env.yaml")


def _autoload_env() -> None:
    """Populate missing FULL_MIX_JUDGE_* vars from runtime_env.yaml.

    Minimal flat-key parse — no yaml dep — since these are all plain strings
    under a single `env_vars:` block.
    """
    if not os.path.exists(_RUNTIME_ENV):
        return
    pat = re.compile(r'^\s+([A-Z_][A-Z0-9_]*):\s*"?([^"#]*?)"?\s*$')
    with open(_RUNTIME_ENV) as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key.startswith("FULL_MIX_JUDGE_") and not os.environ.get(key):
                os.environ[key] = val


def _redact(key: str | None) -> str:
    if not key:
        return "<none>"
    return f"{key[:8]}…{key[-4:]}" if len(key) > 14 else "<short>"


def build_kwargs(args, model: str) -> dict:
    """Reproduce judge_client.call_judge's request payload exactly."""
    messages = [
        {"role": "system", "content": "You are a strict JSON-only grader."},
        {
            "role": "user",
            "content": (
                "Score the answer 'Paris' to the question 'Capital of France?' "
                'Respond ONLY with a JSON object: {"score": <0..1>, "reason": "<text>"}'
            ),
        },
    ]
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    force_json = (not args.no_json) and os.environ.get("FULL_MIX_JUDGE_FORCE_JSON", "1") != "0"
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    disable_thinking = (not args.no_thinking) and os.environ.get(
        "FULL_MIX_JUDGE_DISABLE_THINKING", "1"
    ) != "0"
    if disable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return kwargs


def classify(resp_or_exc) -> tuple[str, str]:
    """Return (bucket, detail). Mirrors judge_client's failure taxonomy."""
    import openai

    if isinstance(resp_or_exc, BaseException):
        e = resp_or_exc
        status = getattr(e, "status_code", None)
        if isinstance(e, openai.APITimeoutError):
            return "timeout", str(e)[:160]
        if isinstance(e, openai.APIConnectionError):
            return "conn_error", str(e)[:160]
        if isinstance(e, openai.RateLimitError):
            return "http_429", str(e)[:160]
        if isinstance(e, openai.InternalServerError):
            return "http_5xx", f"{status}: {str(e)[:140]}"
        if isinstance(e, getattr(openai, "BadRequestError", ())):
            # often: response_format / param unsupported by the provider
            return "http_400_badrequest", str(e)[:200]
        import json as _json

        if isinstance(e, _json.JSONDecodeError):
            return "json_decode", str(e)[:160]
        if isinstance(status, int):
            return f"http_{status}", str(e)[:160]
        return f"other:{type(e).__name__}", str(e)[:160]

    # success-path response object
    resp = resp_or_exc
    choices = getattr(resp, "choices", None)
    if not choices:
        return "no_choices", f"error={getattr(resp, 'error', None)!r}"
    ch0 = choices[0]
    content = (getattr(ch0.message, "content", None) or "")
    finish = getattr(ch0, "finish_reason", None)
    if not content.strip():
        return "empty_content", f"finish_reason={finish!r}"
    return "ok", f"finish_reason={finish!r}"


def fetch_gen_stats(api_base: str, api_key: str, gen_id: str) -> str:
    """Query OpenRouter's /generation endpoint for native provider details."""
    try:
        import json as _json
        import urllib.request

        root = api_base.rstrip("/")
        if root.endswith("/api/v1"):
            url = f"{root}/generation?id={gen_id}"
        else:
            url = f"{root}/generation?id={gen_id}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = _json.loads(r.read()).get("data", {})
        return (
            f"provider={data.get('provider_name')!r} "
            f"native_finish={data.get('native_finish_reason')!r} "
            f"tokens={data.get('tokens_completion')!r}"
        )
    except Exception as e:  # stats are best-effort
        return f"<gen-stats failed: {type(e).__name__}: {e}>"


def main() -> int:
    _autoload_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=int(os.environ.get("DIAG_N", 50)))
    ap.add_argument("-c", "--concurrency", type=int,
                    default=int(os.environ.get("FULL_MIX_JUDGE_CONCURRENCY", 16)))
    ap.add_argument("--model", default=os.environ.get("FULL_MIX_JUDGE_MODEL"))
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("FULL_MIX_JUDGE_MAX_TOKENS", 512)))
    ap.add_argument("--timeout", type=float,
                    default=float(os.environ.get("FULL_MIX_JUDGE_TIMEOUT", 60)))
    ap.add_argument("--no-json", action="store_true", help="drop response_format=json_object")
    ap.add_argument("--no-thinking", action="store_true", help="drop enable_thinking extra_body")
    ap.add_argument("--gen-stats", action="store_true",
                    help="query OpenRouter /generation for provider+native finish reason")
    ap.add_argument("--show-errors", type=int, default=8,
                    help="print up to N example error details per bucket")
    args = ap.parse_args()

    api_base = os.environ.get("FULL_MIX_JUDGE_API_BASE")
    api_key = os.environ.get("FULL_MIX_JUDGE_API_KEY", "EMPTY")
    if not api_base or not args.model:
        print("ERROR: FULL_MIX_JUDGE_API_BASE and FULL_MIX_JUDGE_MODEL must be set "
              "(or present in runtime_env.yaml).", file=sys.stderr)
        return 2

    try:
        import openai
    except ImportError:
        print("ERROR: `pip install openai` first.", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"endpoint   : {api_base}")
    print(f"model      : {args.model}")
    print(f"api_key    : {_redact(api_key)}")
    print(f"requests   : {args.num}   concurrency: {args.concurrency}   "
          f"max_tokens: {args.max_tokens}   timeout: {args.timeout}s")
    print(f"json_object: {not args.no_json}   enable_thinking=False: {not args.no_thinking}")
    print("=" * 72, flush=True)

    client = openai.OpenAI(base_url=api_base, api_key=api_key, timeout=args.timeout)
    kwargs = build_kwargs(args, args.model)

    buckets: collections.Counter = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    latencies: list[float] = []
    gen_ids: list[tuple[str, str]] = []  # (gen_id, bucket)
    lock = threading.Lock()

    def one(_i: int):
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kwargs)
            dt = time.monotonic() - t0
            bucket, detail = classify(resp)
            gid = getattr(resp, "id", None)
            return bucket, detail, dt, gid
        except Exception as e:  # noqa: BLE001 — we classify everything
            dt = time.monotonic() - t0
            bucket, detail = classify(e)
            return bucket, detail, dt, None

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, i) for i in range(args.num)]
        done = 0
        for fut in as_completed(futs):
            bucket, detail, dt, gid = fut.result()
            with lock:
                buckets[bucket] += 1
                latencies.append(dt)
                if len(examples[bucket]) < args.show_errors and bucket != "ok":
                    examples[bucket].append(f"[{dt:5.1f}s] {detail}")
                if gid and bucket != "ok":
                    gen_ids.append((gid, bucket))
                done += 1
            if done % max(1, args.num // 10) == 0:
                print(f"  ...{done}/{args.num}", flush=True)
    wall = time.monotonic() - t_start

    ok = buckets.get("ok", 0)
    n = args.num
    print("\n" + "=" * 72)
    print(f"RESULTS  ({ok}/{n} ok = {100*ok/n:.1f}% success)   wall={wall:.1f}s")
    print("-" * 72)
    for bucket, cnt in buckets.most_common():
        flag = "OK " if bucket == "ok" else "ERR"
        print(f"  [{flag}] {bucket:24s} {cnt:4d}  ({100*cnt/n:5.1f}%)")
    if latencies:
        latencies.sort()
        p = lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))]
        print("-" * 72)
        print(f"  latency  mean={statistics.mean(latencies):.1f}s  "
              f"p50={p(0.5):.1f}s  p90={p(0.9):.1f}s  p99={p(0.99):.1f}s  "
              f"max={latencies[-1]:.1f}s")

    for bucket in buckets:
        if bucket == "ok" or not examples[bucket]:
            continue
        print("-" * 72)
        print(f"  examples — {bucket}:")
        for ex_detail in examples[bucket]:
            print(f"    {ex_detail}")

    if args.gen_stats and gen_ids:
        print("-" * 72)
        print(f"  OpenRouter /generation stats for {min(len(gen_ids), 10)} failed gens:")
        for gid, bucket in gen_ids[:10]:
            print(f"    [{bucket}] {gid}: {fetch_gen_stats(api_base, api_key, gid)}")

    print("=" * 72)
    # Quick verdict to stdout for scripting.
    fail_rate = 100 * (n - ok) / n
    if fail_rate == 0:
        print("VERDICT: endpoint healthy in this sample.")
    else:
        top = [b for b, _ in buckets.most_common() if b != "ok"][:1]
        print(f"VERDICT: {fail_rate:.1f}% failure; dominant mode: {top[0] if top else '?'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
