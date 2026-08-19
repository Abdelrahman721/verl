"""Stats cache for the stage-6 retention pool (consumes stage-5 coding rollouts).

Walks ``/data/abdelrahman/verl/coding_ml_rollouts/*.jsonl``, groups by chat-
templated ``input``, computes per-prompt mean ``score``, and matches each
prompt against the stage-5 training source (the 35 120 coding rows in
``data/medical_qa_stage5/train.parquet``).

Output: a JSON cache the stage-6 retention build consumes to identify
"unmastered" stage-5 coding prompts (mean_score < threshold).

Reuses verbatim:
  - render_chat / build_rendered_index / _hash16 from stats_medical_qa_next.py

Usage:
    python -m full_mix.curriculum.stats_medical_qa_stage6_retention \\
        --rollout_dir   /data/abdelrahman/verl/coding_ml_rollouts \\
        --coding_source /data/abdelrahman/verl/data/medical_qa_stage5/train.parquet \\
        --tokenizer     /data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh/global_step_500/merged_hf_model \\
        --out           /data/abdelrahman/verl/data/medical_qa_stage6/_stage5_retention_cache.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from full_mix.curriculum.stats_medical_qa_next import (  # noqa: E402
    build_rendered_index,
    render_chat,
)


# Stage-5 coding eval_modes (the four task_types the trainer emits in
# `reward/eval_mode`). A prompt counts as "stage-5 coding" iff every observed
# eval_mode is in this set — defensive against a prompt being scored under
# multiple labels across steps.
_CODING_EVAL_MODES = frozenset({
    "icd10_multilabel",
    "icd10_instruction_follow",
    "snomed_multilabel",
    "snomed_instruction_follow",
})


def _hash16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _render_coding(tok, coding_parquet: str) -> tuple[dict[str, int], int]:
    """Return (rendered_to_source_idx, n_source_rows)."""
    import pyarrow.parquet as pq
    t = pq.read_table(coding_parquet, columns=["prompt"])
    prompts = t.column("prompt").to_pylist()
    rendered = [render_chat(tok, p) for p in prompts]
    rendered_to_idx, collisions = build_rendered_index(rendered)
    print(f"  coding rendered: {len(rendered)} rows  → distinct keys: "
          f"{len(rendered_to_idx)} (collisions={collisions})")
    return rendered_to_idx, len(rendered)


def _walk_rollouts(rollout_dir: str) -> tuple[int, int, list[dict]]:
    """Walk every rollout row and group by `input`. Returns
    (n_total, n_kept, per_prompt)."""
    by_input: dict[str, dict] = {}
    n_total = 0
    n_kept = 0
    files = sorted(glob.glob(os.path.join(rollout_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no *.jsonl files under {rollout_dir}")
    for fp in files:
        try:
            step = int(os.path.basename(fp).split(".")[0])
        except ValueError:
            step = -1
        with open(fp) as f:
            for line in f:
                n_total += 1
                r = json.loads(line)
                inp = r.get("input")
                if not inp:
                    continue
                n_kept += 1
                em = r.get("reward/eval_mode") or ""
                s = by_input.get(inp)
                if s is None:
                    s = by_input[inp] = {"scores": [], "em": set(), "steps": set()}
                s["scores"].append(float(r["score"]))
                if em:
                    s["em"].add(em)
                s["steps"].add(step)
    per_prompt = [
        {
            "input":        inp,
            "mean_score":   sum(d["scores"]) / len(d["scores"]),
            "n_samples":    len(d["scores"]),
            "n_steps_seen": len(d["steps"]),
            "eval_mode":    sorted(d["em"]),
        }
        for inp, d in by_input.items()
    ]
    return n_total, n_kept, per_prompt


def _is_coding_prompt(em_list: list[str]) -> bool:
    return bool(em_list) and all(em in _CODING_EVAL_MODES for em in em_list)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rollout_dir",   required=True)
    ap.add_argument("--coding_source", required=True,
                    help="data/medical_qa_stage5/train.parquet")
    ap.add_argument("--tokenizer",     required=True)
    ap.add_argument("--out",           required=True)
    ap.add_argument("--thresholds",    default="0.5,0.6,0.7,0.8")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    print(f"rollout_dir   : {args.rollout_dir}")
    print(f"coding_source : {args.coding_source}")
    print(f"tokenizer     : {args.tokenizer}")

    # ----- Phase 1: walk rollouts -----
    print("\n=== Phase 1: walk stage-5 coding rollouts ===")
    t0 = time.time()
    n_total, n_kept, per_prompt = _walk_rollouts(args.rollout_dir)
    em_counts = Counter(",".join(p["eval_mode"]) for p in per_prompt)
    print(f"  total rollout rows: {n_total}   kept (input present): {n_kept}")
    print(f"  distinct rollout inputs: {len(per_prompt)}")
    print(f"  eval_mode breakdown (distinct prompts):")
    for em, c in sorted(em_counts.items()):
        print(f"    {em!r:<48} {c}")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ----- Phase 2: render coding source -----
    print("\n=== Phase 2: render stage-5 coding source ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.time()
    coding_to_src, n_coding_rows = _render_coding(tok, args.coding_source)
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ----- Phase 3: match rollouts → source -----
    print("\n=== Phase 3: match rollouts → coding source rows ===")
    n_matched = 0
    n_orphans = 0
    seen_coding_idxs: set[int] = set()
    for p in per_prompt:
        ci = coding_to_src.get(p["input"])
        p["coding_source_index"] = ci
        if ci is not None:
            n_matched += 1
            seen_coding_idxs.add(ci)
        else:
            n_orphans += 1
    print(f"  coding-matched: {n_matched}  distinct coding rows seen: "
          f"{len(seen_coding_idxs)} / {n_coding_rows}")
    print(f"  orphans (general/medical retention from stage-5 retention pool): {n_orphans}")

    # ----- Phase 4: threshold what-ifs over coding pool -----
    print("\n=== Phase 4: threshold what-ifs (coding pool only) ===")
    coding_pp = [p for p in per_prompt
                 if p["coding_source_index"] is not None and _is_coding_prompt(p["eval_mode"])]
    print(f"  coding prompts with matched source AND pure-coding eval_modes: {len(coding_pp)}")
    for thr in thresholds:
        carry = sum(1 for p in coding_pp if p["mean_score"] <  thr)
        mast  = sum(1 for p in coding_pp if p["mean_score"] >= thr)
        print(f"  threshold={thr:>5.2f}:  unmastered={carry:>5}   mastered={mast:>5}")
    if coding_pp:
        scores = [p["mean_score"] for p in coding_pp]
        print(f"  coding mean_score across seen: {statistics.fmean(scores):.4f}")
        qs = statistics.quantiles(scores, n=100) if len(scores) >= 100 else None
        if qs:
            print(f"  percentiles: P10={qs[9]:.3f} P50={qs[49]:.3f} P90={qs[89]:.3f}")

    # ----- Phase 5: emit cache -----
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    seen_coding_strict = sorted({
        p["coding_source_index"] for p in per_prompt
        if p["coding_source_index"] is not None and _is_coding_prompt(p["eval_mode"])
    })

    cache = {
        "meta": {
            "rollout_dir":           args.rollout_dir,
            "coding_source_parquet": args.coding_source,
            "tokenizer":             args.tokenizer,
            "n_rollout_rows":        n_total,
            "n_kept_rollout_rows":   n_kept,
            "n_distinct_prompts":    len(per_prompt),
            "n_coding_matched":      n_matched,
            "n_orphans":             n_orphans,
            "n_coding_rows_in_source": n_coding_rows,
            "thresholds_reported":   thresholds,
        },
        "per_prompt": [
            {
                "input_hash":          _hash16(p["input"]),
                "mean_score":          round(p["mean_score"], 4),
                "n_samples":           p["n_samples"],
                "n_steps_seen":        p["n_steps_seen"],
                "eval_mode":           p["eval_mode"],
                "coding_source_index": p["coding_source_index"],
            }
            for p in per_prompt
        ],
        "seen_coding_source_indices_strict": seen_coding_strict,
        "seen_coding_source_indices_any":    sorted(seen_coding_idxs),
    }
    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
