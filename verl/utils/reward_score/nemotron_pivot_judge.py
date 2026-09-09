# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Judge-augmented reward for data_source "nemotron_pivot".

Same task and the same tool-call scoring as `nemotron_pivot`; the ONLY change is how a
row whose expert action was prose is scored.

  * expected TOOL CALL -> smooth score in [-1, 1] (see smooth_call_score below).
  * expected PROSE     -> if the policy emitted a parsable tool call, -1.0 without calling
                          the judge (a modality error, mirroring prose-where-a-call-was-
                          expected). Otherwise an LLM judge asks only whether the reply is
                          coherent prose addressing the last user message; the gold reply
                          is NOT shown, so this is a leniency gate, not a correctness check.
                          Its verdict maps to +1.0 (pass) / -1.0 (fail) so prose and tool
                          rows share the same [-1, 1] range.

Why: under the rule reward, ANY prose scores 1.0 when the expert replied in prose
(content is never compared), so the policy collapses onto prose. The judge prices prose
on substance instead.

The judge is minimax/minimax-m3 via OpenRouter. Set OPENROUTER_API_KEY. Tunables:
  NEMOTRON_JUDGE_MODEL     (default minimax/minimax-m3)
  NEMOTRON_JUDGE_PROVIDER  (default minimax/fp8; empty string = let OpenRouter route)
  NEMOTRON_JUDGE_EFFORT    (default high; low is ~2x cheaper but measurably less stable)
  NEMOTRON_JUDGE_TIMEOUT   (default 120 seconds)
  NEMOTRON_JUDGE_RETRIES   (default 6; 429s are common, see below)
  NEMOTRON_JUDGE_FALLBACK  (default 0.0) score used when the judge cannot be reached.
                           0.0 is the neutral midpoint of the [-1, 1] prose range, so an
                           outage distorts a GRPO group least. Set it to -1 to treat an
                           unreachable judge as a failure instead.
  NEMOTRON_JUDGE_MAX_CHARS (default 6000) per-field truncation sent to the judge

Provider choice matters a lot. Pinning DeepInfra (the previous default) returned HTTP 429
with provider_error_code=engine_overloaded and limit_source=upstream_provider_shared_pool
on ~90% of requests at concurrency 4 - that pool is shared across all OpenRouter traffic,
so it congests independently of how hard we push it, and no Retry-After header is sent, so
the backoff below is blind. Measured over 16 training steps that cost 7-22% of judge calls,
which fell back to the neutral score. The provider is therefore pinned to minimax/fp8, the
first-party MiniMax endpoint, which served 20/20 at concurrency 4 in the same test. The pin
is exclusive (allow_fallbacks=False) so every verdict comes from identical serving weights.
Set NEMOTRON_JUDGE_PROVIDER="" to let OpenRouter route across all providers, which raises
the throughput ceiling at some cost in verdict stability. 429 is still retried with
exponential backoff honouring Retry-After, and NEMOTRON_JUDGE_WORKERS defaults to 8.

On judge failure the fallback is 0.5 rather than 0.0 or 1.0: under GRPO a neutral score
distorts the group's relative advantage least, and judge_error is logged so outages are
visible in the metrics rather than silently biasing toward or against prose.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from . import nemotron_pivot

_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = """A user said this to a customer-support assistant:

<user_message>
{user}
</user_message>

The assistant replied:

<candidate>
{cand}
</candidate>

Is the candidate a coherent prose reply that addresses what the user said?

Judge only the reply itself - there is no reference answer, and you are not
checking whether its facts are correct. Answer 1 unless the reply is genuinely
broken: empty, cut off mid-sentence, incoherent or repetitive, or plainly about
something other than what the user asked.

1 - coherent prose that addresses the user
0 - broken, incoherent, or unrelated

End your reply with exactly one line:
SCORE: <1 or 0>"""

def _smooth_one(exp: dict, act: dict) -> float:
    """Score one expected call against one emitted call, in [-1, 1].

      name mismatch                      -> -1, scored no further
      expected key present in the call   -> +1
      that key's value also correct      -> +1   (wrong value -> -1)
      key emitted that is not expected   -> -1
      expected key missing entirely      ->  0   (it simply earns none of its 2 points)

    Normalised by the ground truth's own maximum, 2 points per expected argument, so a
    perfect call is +1 and a name match with everything wrong lands at -1. Value equality
    uses nemotron_pivot._arguments_match, so the fuzzy string rules are unchanged.
    """
    if exp["name"] != act["name"]:
        return -1.0
    try:
        ea = json.loads(exp["arguments"])
        aa = json.loads(act["arguments"])
    except (json.JSONDecodeError, TypeError):
        return -1.0
    if not isinstance(ea, dict) or not isinstance(aa, dict):
        return 1.0 if ea == aa else -1.0

    pts = 0
    for k, v in ea.items():
        if k in aa:
            pts += 1
            pts += 1 if nemotron_pivot._arguments_match(v, aa[k]) else -1
    for k in aa:
        if k not in ea:
            pts -= 1

    # A tool that takes no arguments: there are no points to earn, so the old
    # `denom = max(2*len(ea), 1)` scored a PERFECT zero-arg call 0.0 instead of 1.0.
    # 2.3% of calls in the unified corpus are zero-arg, and 1.4% of tool rows are
    # entirely zero-arg. Emitting nothing extra is exactly right; extra keys are
    # still charged at -1 each.
    if not ea:
        return 1.0 if not aa else max(-1.0, float(-len(aa)))

    denom = 2 * len(ea)
    return max(-1.0, min(1.0, pts / denom))


def smooth_call_score(expected: dict, actual: dict) -> float:
    """Whole-response tool score in [-1, 1]. Prose where a call was expected is -1.

    Emitted calls are matched to ground-truth calls BY FUNCTION NAME, respecting
    multiplicity: if the ground truth calls `f` twice, two emitted `f` calls are both
    matched, a third is an extra. Within a name group each expected call is paired with
    its best unused emitted call, so ordering never matters.

      matched call      -> its _smooth_one score, in [-1, 1], summed
      extra call        -> -1 each. "Extra" means a name absent from the ground truth,
                           or a repeat beyond the ground truth's own multiplicity.
      expected call the
      model never made  -> -1 each, symmetric with an extra call

    Normalised by the number of ground-truth calls, so a perfect response is +1, and
    clamped so heavy over- or under-calling cannot push below -1. Omission and
    commission cost the same: emitting 1 of 3 expected calls scores (+1-1-1)/3.

    This replaces an earlier hard gate that returned -1 whenever the emitted count
    differed from the ground truth's. That gate made "4 of 5 calls correct" score
    exactly what prose scored, leaving no gradient to climb on multi-call rows - half
    the tool rows in the unified corpus need more than one call.
    """
    exp_calls = nemotron_pivot._calls_of(expected)
    act_calls = nemotron_pivot._calls_of(actual)
    if not exp_calls:
        return -1.0
    # No guard needed for prose: every expected call goes unmade, so the sum is
    # -len(exp_calls) and the score is exactly -1.

    exp_by: dict[str, list] = {}
    act_by: dict[str, list] = {}
    for e in exp_calls:
        exp_by.setdefault(e["name"], []).append(e)
    for a in act_calls:
        act_by.setdefault(a["name"], []).append(a)

    total = 0.0
    extras = 0
    for name, es in exp_by.items():
        cands = list(act_by.get(name, []))
        used: set[int] = set()
        for e in es:
            best, best_j = None, None
            for j, a in enumerate(cands):
                if j in used:
                    continue
                v = _smooth_one(e, a)
                if best is None or v > best:
                    best, best_j = v, j
            if best_j is None:
                total -= 1.0  # expected call never made: charged like an extra
                continue
            used.add(best_j)
            total += best
        extras += max(0, len(cands) - len(es))
    for name, as_ in act_by.items():
        if name not in exp_by:
            extras += len(as_)

    return max(-1.0, min(1.0, (total - extras) / len(exp_calls)))


_VALID = {"1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}

_cache: dict[str, float] = {}
_cache_lock = threading.Lock()


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v


def _parse_score(text: str) -> float | None:
    """Only an explicit trailing SCORE: line counts. Bare numbers inside the model's
    deliberation are not a verdict and reading them produced spurious 0.5s in testing."""
    tail = text.rsplit("SCORE:", 1)
    if len(tail) != 2:
        return None
    tok = tail[1].strip().split()[0].strip().rstrip(".,`*") if tail[1].strip() else ""
    return _VALID.get(tok)


def judge_prose(user_msg: str, candidate: str) -> tuple[float | None, str]:
    """-> (score in {1.0, 0.0}, error string). score is None on failure.

    The gold reply is deliberately NOT shown to the judge: prose is scored on coherence
    and relevance to the user's message alone, making this a lenient gate rather than a
    correctness check."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None, "OPENROUTER_API_KEY not set"

    n = int(_env("NEMOTRON_JUDGE_MAX_CHARS", "6000"))
    prompt = PROMPT.format(user=user_msg[:n], cand=candidate[:n] or "(empty reply)")

    h = hashlib.sha1(prompt.encode()).hexdigest()
    with _cache_lock:
        if h in _cache:
            return _cache[h], ""

    body: dict[str, Any] = {
        "model": _env("NEMOTRON_JUDGE_MODEL", "minimax/minimax-m3"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
        "temperature": 0,
        "reasoning": {"effort": _env("NEMOTRON_JUDGE_EFFORT", "high")},
    }
    provider = _env("NEMOTRON_JUDGE_PROVIDER", "minimax/fp8")
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}

    timeout = float(_env("NEMOTRON_JUDGE_TIMEOUT", "120"))
    retries = int(_env("NEMOTRON_JUDGE_RETRIES", "6"))
    last = ""
    backoff = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                _URL,
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            msg = payload["choices"][0]["message"]
            score = _parse_score(msg.get("content") or "")
            if score is not None:
                with _cache_lock:
                    _cache[h] = score
                return score, ""
            last = f"unparsable verdict (finish_reason={payload['choices'][0].get('finish_reason')})"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 429:
                # Respect Retry-After when the provider sends one, else exponential backoff.
                try:
                    ra = float(e.headers.get("Retry-After") or 0)
                except (TypeError, ValueError):
                    ra = 0.0
                if attempt < retries - 1:
                    time.sleep(max(ra, backoff) + random.uniform(0, 1))
                    backoff = min(backoff * 2, 60.0)
                continue
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(backoff + random.uniform(0, 1))
            backoff = min(backoff * 2, 60.0)
    return None, last


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    last_user_message: str = "",
    **kwargs: Any,
) -> dict:
    """Drop-in for nemotron_pivot.compute_score, plus `last_user_message`.

    Extra logged keys over the rule reward: judged, judge_score, judge_error.
    """
    base = nemotron_pivot.compute_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)
    base.setdefault("judged", 0.0)
    base.setdefault("judge_score", 0.0)
    base.setdefault("judge_error", 0.0)

    try:
        expected = json.loads(ground_truth or "")
    except json.JSONDecodeError:
        return base
    if expected.get("type") != "message":
        if _env("NEMOTRON_SMOOTH_CALLS", "1") == "1":
            actual = nemotron_pivot.extract_action(solution_str or "")
            base["score"] = smooth_call_score(expected, actual)
        return base

    # Policy emitted a parsable tool call where prose was expected: -1.0, no judge call.
    # Symmetric with smooth_call_score returning -1.0 for prose where a call was expected.
    if base["is_call"]:
        base["score"] = -1.0
        return base

    # What the user would actually read: everything after the final </think>. A response
    # that never closed its think block is judged whole, which is the intent - it is not
    # a usable reply.
    reply = solution_str.rsplit("</think>", 1)[-1].strip() if "</think>" in solution_str else solution_str.strip()

    score, err = judge_prose(last_user_message, reply)
    base["judged"] = 1.0
    if score is None:
        base["judge_error"] = 1.0
        base["score"] = float(_env("NEMOTRON_JUDGE_FALLBACK", "0.0"))
        base["judge_score"] = base["score"]
        if os.getenv("NEMOTRON_JUDGE_VERBOSE"):
            print(f"[nemotron_judge] judge failed: {err}")
        return base
    # judge_score keeps the raw 0/1 verdict for diagnostics; score is mapped onto the
    # same [-1, 1] range the tool reward uses, so the two halves are directly comparable
    # and a GRPO group mixing them is not skewed by scale.
    base["judge_score"] = score
    base["score"] = 1.0 if score == 1.0 else -1.0
    return base
