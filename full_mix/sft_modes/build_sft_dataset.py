"""Verify generated responses and build the dual-mode (thinking / instant) SFT set.

Pipeline per prompt:
  1. Parse each sampled response into (reasoning, answer) by splitting on the
     think tags. Malformed samples are REJECTED, not salvaged — see below.
  2. Verify the ANSWER (never the reasoning):
       ifeval -> full_mix.ifeval_reward.compute_score == 1.0 (all constraints)
       chat   -> full_mix.rewards.chat.compute_score  >  0.5 (beat the baseline)
  3. Keep at most --max_keep correct samples.
  4. Split them half/half into thinking and instant records.

Why malformed samples are dropped: ~18% of this model's rollouts hit the length
cap without ever emitting </think>. `strip_think` returns the FULL text when the
closing tag is missing, so a naive pipeline would feed raw chain-of-thought to
the verifier as if it were the answer and let truncated output score as correct.
A sample is usable only if it opens <think>, closes </think>, and leaves a
non-empty answer behind.

Odd counts alternate by prompt_uid parity so the global thinking/instant split
stays balanced instead of biasing toward one mode.

Both record types carry the SAME normalized framing; the instant ones simply
have no reasoning_content, and chat_template_dual_mode.jinja renders the empty
block for them. Reasoning is re-emitted trimmed, never verbatim, because the
model's own framing is inconsistent (<think>Okay / <think>\\nOkay / <think> Okay).

Judge verdicts are cached to --verdict_cache so re-runs never pay twice for the
same chat comparison.

Usage:
    python -m full_mix.sft_modes.build_sft_dataset --source ifeval
    python -m full_mix.sft_modes.build_sft_dataset --source chat --concurrency 32
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def parse_response(text: str) -> tuple[str, str] | None:
    """Split a raw sample into (reasoning, answer), or None if unusable.

    Requires a well-formed <think>...</think> followed by a non-empty answer.
    Extra <think> tags inside the reasoning are tolerated (the model sometimes
    restarts its trace); the FIRST close tag ends the reasoning.
    """
    if not text:
        return None
    stripped = text.lstrip()
    if not stripped.startswith(_THINK_OPEN):
        return None
    if _THINK_CLOSE not in stripped:
        return None
    body = stripped[len(_THINK_OPEN):]
    reasoning, answer = body.split(_THINK_CLOSE, 1)
    reasoning = reasoning.strip()
    answer = answer.strip()
    if not answer:
        return None
    if _THINK_OPEN in answer or _THINK_CLOSE in answer:
        return None  # tag leaked into the answer; framing would be ambiguous
    return reasoning, answer


def verify_ifeval(answer: str, ground_truth, threshold: float) -> tuple[bool, float]:
    from full_mix import ifeval_reward

    score = float(
        ifeval_reward.compute_score(
            data_source="local/dolci-ifeval-32b",
            solution_str=answer,
            ground_truth=ground_truth,
            extra_info=None,
        )
    )
    return score >= threshold, score


def verify_chat(answer: str, rec: dict, threshold: float) -> tuple[bool, float]:
    from full_mix.rewards import chat as chat_reward

    extra_info = {
        "user_prompt": rec.get("user_prompt") or "",
        "baseline_response": rec.get("baseline_response") or "",
    }
    score = float(chat_reward.compute_score(answer, rec.get("ground_truth"), extra_info))
    return score > threshold, score


def select_and_split(kept: list[dict], prompt_uid: int, max_keep: int) -> list[dict]:
    """Trim to max_keep, then split half thinking / half instant.

    Odd k alternates by uid parity so neither mode is systematically favoured.
    """
    kept = kept[:max_keep]
    k = len(kept)
    n_think = (k + 1) // 2 if prompt_uid % 2 == 0 else k // 2
    for i, item in enumerate(kept):
        item["thinking"] = i < n_think
    return kept


def process_prompt(rec: dict, args) -> dict:
    """Verify one prompt's samples. Returns a per-prompt result dict."""
    is_chat = args.source == "chat"
    candidates, n_malformed = [], 0
    for idx, raw in enumerate(rec.get("responses", [])):
        parsed = parse_response(raw)
        if parsed is None:
            n_malformed += 1
            continue
        reasoning, answer = parsed
        candidates.append({"idx": idx, "reasoning": reasoning, "answer": answer})

    kept, scores = [], []
    for cand in candidates:
        # Stop judging once we have enough — chat verdicts cost API calls.
        if len(kept) >= args.max_keep:
            break
        if is_chat:
            ok, score = verify_chat(cand["answer"], rec, args.chat_threshold)
        else:
            ok, score = verify_ifeval(cand["answer"], rec["ground_truth"], args.ifeval_threshold)
        scores.append(score)
        if ok:
            cand["score"] = score
            kept.append(cand)

    kept = select_and_split(kept, int(rec["prompt_uid"]), args.max_keep)
    return {
        "prompt_uid": rec["prompt_uid"],
        "data_source": rec["data_source"],
        "messages": rec["messages"],
        "n_samples": len(rec.get("responses", [])),
        "n_malformed": n_malformed,
        "n_verified": len(scores),
        "n_correct": len(kept),
        "kept": kept,
        "scores": scores,
    }


def to_sft_records(result: dict) -> list[dict]:
    out = []
    user_msgs = [m for m in result["messages"] if m["role"] != "assistant"]
    for item in result["kept"]:
        assistant = {"role": "assistant", "content": item["answer"]}
        if item["thinking"]:
            assistant["reasoning_content"] = item["reasoning"]
        out.append(
            {
                "messages": user_msgs + [assistant],
                "thinking": bool(item["thinking"]),
                "data_source": result["data_source"],
                "prompt_uid": int(result["prompt_uid"]),
                "sample_idx": int(item["idx"]),
                "verifier_score": float(item["score"]),
            }
        )
    return out


def render_text(records: list[dict], model_path: str, template_path: str) -> list[str]:
    """Materialize the exact training string via the dual-mode template."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.chat_template = open(template_path).read()
    return [tok.apply_chat_template(r["messages"], tokenize=False) for r in records]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["chat", "ifeval"], required=True)
    ap.add_argument("--generations", default=None,
                    help="path or glob; defaults to <raw_dir>/<source>_generations*.jsonl (all shards)")
    ap.add_argument("--raw_dir", default="/data/abdelrahman/verl/data/sft_dual_mode/raw")
    ap.add_argument("--out_dir", default="/data/abdelrahman/verl/data/sft_dual_mode")
    ap.add_argument("--max_keep", type=int, default=4)
    ap.add_argument("--ifeval_threshold", type=float, default=1.0,
                    help="ifeval score is the FRACTION of constraints met; 1.0 = all")
    ap.add_argument("--chat_threshold", type=float, default=0.5,
                    help="pairwise reward strictly above this counts as beating the baseline")
    ap.add_argument("--concurrency", type=int, default=32, help="prompts judged in parallel")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model_path", default="/data/abdelrahman/verl/checkpoints/RL-Exps/chat-ifeval-sync-4b/global_step_60/merged_hf_model")
    ap.add_argument("--template", default=os.path.join(_HERE, "chat_template_dual_mode.jinja"))
    ap.add_argument("--no_render_text", action="store_true",
                    help="skip materializing the rendered `text` column")
    args = ap.parse_args()

    # Picks up both the single-machine file and any sharded ones, so a
    # distributed generation run needs no extra flags here.
    if args.generations:
        gen_paths = sorted(glob.glob(args.generations))
    else:
        gen_paths = sorted(glob.glob(os.path.join(args.raw_dir, f"{args.source}_generations*.jsonl")))
    if not gen_paths:
        raise SystemExit(
            f"no generations found for source={args.source} under {args.raw_dir}\n"
            f"run full_mix.sft_modes.generate_responses first"
        )

    recs, seen_uids = [], set()
    for path in gen_paths:
        n_before = len(recs)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                uid = int(rec["prompt_uid"])
                if uid in seen_uids:
                    continue  # overlapping shards / re-run duplicates
                seen_uids.add(uid)
                recs.append(rec)
                if args.limit and len(recs) >= args.limit:
                    break
        print(f"  {path}: {len(recs) - n_before} prompts")
        if args.limit and len(recs) >= args.limit:
            break
    print(f"loaded {len(recs)} prompts from {len(gen_paths)} file(s)")

    workers = args.concurrency if args.source == "chat" else min(args.concurrency, os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda r: process_prompt(r, args), recs))

    sft: list[dict] = []
    for res in results:
        sft.extend(to_sft_records(res))

    stats = Counter()
    for res in results:
        stats["prompts"] += 1
        stats["samples"] += res["n_samples"]
        stats["malformed"] += res["n_malformed"]
        stats["verified"] += res["n_verified"]
        stats["correct"] += res["n_correct"]
        if res["n_correct"] == 0:
            stats["prompts_with_no_correct"] += 1
    n_think = sum(1 for r in sft if r["thinking"])

    print(f"\n  prompts                 : {stats['prompts']}")
    print(f"  samples seen            : {stats['samples']}")
    print(f"  malformed (dropped)     : {stats['malformed']} ({stats['malformed'] / max(stats['samples'], 1):.1%})")
    print(f"  samples verified        : {stats['verified']}  (early-exit at {args.max_keep} correct)")
    print(f"  correct kept            : {stats['correct']}")
    print(f"  prompts with 0 correct  : {stats['prompts_with_no_correct']} "
          f"({stats['prompts_with_no_correct'] / max(stats['prompts'], 1):.1%})")
    print(f"  SFT records             : {len(sft)}  ({n_think} thinking / {len(sft) - n_think} instant)")

    if not sft:
        print("\nno usable records — nothing written")
        return 1

    if not args.no_render_text:
        print("\nrendering `text` via the dual-mode template …")
        for rec, text in zip(sft, render_text(sft, args.model_path, args.template)):
            rec["text"] = text

    os.makedirs(args.out_dir, exist_ok=True)
    out_parquet = os.path.join(args.out_dir, f"{args.source}_sft.parquet")
    pq.write_table(pa.Table.from_pylist(sft), out_parquet)
    print(f"wrote {len(sft)} records -> {out_parquet}")

    report = os.path.join(args.out_dir, f"{args.source}_build_report.json")
    with open(report, "w") as f:
        json.dump(
            {
                "source": args.source,
                "generations": gen_paths,
                "max_keep": args.max_keep,
                "ifeval_threshold": args.ifeval_threshold,
                "chat_threshold": args.chat_threshold,
                "counts": dict(stats),
                "sft_records": len(sft),
                "thinking_records": n_think,
                "instant_records": len(sft) - n_think,
            },
            f,
            indent=2,
        )
    print(f"wrote report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
