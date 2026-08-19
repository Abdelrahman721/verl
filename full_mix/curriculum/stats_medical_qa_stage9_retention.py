"""Stats cache: stage-2 medical_qa rollouts re-matched to medical_qa source indices.

Walks ``/data/abdelrahman/verl/medical_qa_stage2_rollouts/*.jsonl``, groups
by chat-templated ``input``, and matches each prompt against
``data/medical_qa/train.parquet`` (316 287 rows). Output is a JSON cache
the stage-9 retention build consumes to compute the unseen-vs-stage-1+2
pool.

Why this exists separately from `_stats_cache.json["stage2"]`:
  - The existing stage-2 block indexes a different source parquet
    (`medical_qa_stage2/train.parquet`, ~14 875 rows), so its
    `seen_source_indices` are not directly comparable to stage-1's
    indices into `medical_qa/train.parquet`.
  - Stage 9 wants the UNION of stage-1 + stage-2 seen sets against the
    SAME source (`medical_qa/train.parquet`), so we re-walk stage-2
    rollouts and match against that source.

Reuses verbatim:
  - render_chat / build_rendered_index from stats_medical_qa_next.py

Usage:
    python -m full_mix.curriculum.stats_medical_qa_stage9_retention \\
        --rollout_dir    /data/abdelrahman/verl/medical_qa_stage2_rollouts \\
        --medical_source /data/abdelrahman/verl/data/medical_qa/train.parquet \\
        --tokenizer      /data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh/global_step_500/merged_hf_model \\
        --out            /data/abdelrahman/verl/data/medical_qa_stage9/_stage2_medqa_seen_cache.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
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


def _hash16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _render_medqa(tok, medical_parquet: str) -> tuple[dict[str, int], int]:
    """Render every prompt in medical_qa/train.parquet via chat template;
    return (rendered_to_source_idx, n_source_rows). 316k rows; expect a
    few minutes of CPU."""
    import pyarrow.parquet as pq
    t = pq.read_table(medical_parquet, columns=["prompt"])
    prompts = t.column("prompt").to_pylist()
    rendered = [render_chat(tok, p) for p in prompts]
    rendered_to_idx, collisions = build_rendered_index(rendered)
    print(f"  medical rendered: {len(rendered)} rows  → distinct keys: "
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rollout_dir",    required=True)
    ap.add_argument("--medical_source", required=True,
                    help="data/medical_qa/train.parquet (316 287 rows)")
    ap.add_argument("--tokenizer",      required=True)
    ap.add_argument("--out",            required=True)
    args = ap.parse_args()

    print(f"rollout_dir    : {args.rollout_dir}")
    print(f"medical_source : {args.medical_source}")
    print(f"tokenizer      : {args.tokenizer}")

    # ----- Phase 1: walk rollouts -----
    print("\n=== Phase 1: walk stage-2 medical_qa rollouts ===")
    t0 = time.time()
    n_total, n_kept, per_prompt = _walk_rollouts(args.rollout_dir)
    em_counts = Counter(",".join(p["eval_mode"]) for p in per_prompt)
    print(f"  total rollout rows: {n_total}   kept (input present): {n_kept}")
    print(f"  distinct rollout inputs: {len(per_prompt)}")
    print(f"  eval_mode breakdown (distinct prompts):")
    for em, c in sorted(em_counts.items()):
        print(f"    {em!r:<48} {c}")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ----- Phase 2: render medical_qa source -----
    print("\n=== Phase 2: render medical_qa source ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.time()
    medqa_to_src, n_medqa_rows = _render_medqa(tok, args.medical_source)
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ----- Phase 3: match rollouts → medical_qa source -----
    print("\n=== Phase 3: match rollouts → medical_qa source rows ===")
    n_matched = 0
    n_orphans = 0
    seen_medqa_idxs: set[int] = set()
    for p in per_prompt:
        ci = medqa_to_src.get(p["input"])
        p["medqa_source_index"] = ci
        if ci is not None:
            n_matched += 1
            seen_medqa_idxs.add(ci)
        else:
            n_orphans += 1
    print(f"  matched: {n_matched}  distinct medqa rows seen: "
          f"{len(seen_medqa_idxs)} / {n_medqa_rows}")
    print(f"  orphans (not present in medical_qa/train.parquet): {n_orphans}")

    # ----- Phase 4: emit cache -----
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cache = {
        "meta": {
            "rollout_dir":           args.rollout_dir,
            "medical_source_parquet": args.medical_source,
            "tokenizer":             args.tokenizer,
            "n_rollout_rows":        n_total,
            "n_kept_rollout_rows":   n_kept,
            "n_distinct_prompts":    len(per_prompt),
            "n_matched":             n_matched,
            "n_orphans":             n_orphans,
            "n_medqa_rows_in_source": n_medqa_rows,
        },
        "per_prompt": [
            {
                "input_hash":         _hash16(p["input"]),
                "mean_score":         round(p["mean_score"], 4),
                "n_samples":          p["n_samples"],
                "n_steps_seen":       p["n_steps_seen"],
                "eval_mode":          p["eval_mode"],
                "medqa_source_index": p["medqa_source_index"],
            }
            for p in per_prompt
        ],
        "seen_medqa_source_indices": sorted(seen_medqa_idxs),
    }
    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
