"""Stats for the medical-benchmark retention pool (stage-4 layer).

Walks `/data/abdelrahman/verl/medical_qa_benchmark_rollouts/*.jsonl`, groups by
chat-templated `input`, computes per-prompt mean `score`, matches each rollout
input back to its source row in `data/medical_qa_stage3/train.parquet`
(filtered to medical_benchmark_* rows), and emits a cache the retention
build script consumes.

Why this exists separately from stats_medical_qa_stage3.py:
  - The benchmark rollouts dir is NEW (came from a later run) and doesn't
    overlap with the stage-1/stage-2 rollouts already cached.
  - We only care about bench rows in stage-3 source, so the renderer is
    restricted to those — much faster than rendering all 21 193 rows.

Reuses verbatim:
  - render_chat / walk_rollouts / aggregate_rollouts / build_rendered_index
    from stats_medical_qa_stage3.py.

Usage:
    python -m full_mix.curriculum.stats_medical_qa_bench_retention \\
        --rollout_dir /data/abdelrahman/verl/medical_qa_benchmark_rollouts \\
        --source      /data/abdelrahman/verl/data/medical_qa_stage3/train.parquet \\
        --tokenizer   /data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh/global_step_500/merged_hf_model \\
        --out         /data/abdelrahman/verl/data/medical_qa_stage4/_bench_retention_cache.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict


_BENCH_DATA_SOURCES = frozenset({
    "medical_benchmark_mcq",
    "medical_benchmark_numeric",
    "medical_benchmark_medec",
    "medical_benchmark_text",
})


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from full_mix.curriculum.stats_medical_qa_next import (  # noqa: E402
    aggregate_rollouts,
    build_rendered_index,
    render_chat,
)


def _hash16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def render_bench_rows(tok, parquet_path: str) -> tuple[list[str], list[str], list[int]]:
    """Load source parquet, keep only medical_benchmark_* rows, render each.

    Returns:
        rendered[i]: the chat-template-rendered string of bench row #i
        data_sources[i]: corresponding data_source label
        bench_source_indices[i]: the ORIGINAL row index in the source parquet
    """
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path, columns=["prompt", "data_source"])
    prompts = t.column("prompt").to_pylist()
    data_sources = t.column("data_source").to_pylist()
    rendered, ds_kept, idx_kept = [], [], []
    for i, (p, ds) in enumerate(zip(prompts, data_sources)):
        if ds not in _BENCH_DATA_SOURCES:
            continue
        rendered.append(render_chat(tok, p))
        ds_kept.append(ds)
        idx_kept.append(i)
    return rendered, ds_kept, idx_kept


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rollout_dir", required=True)
    ap.add_argument("--source", required=True,
                    help="data/medical_qa_stage3/train.parquet")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    print(f"rollout_dir: {args.rollout_dir}")
    print(f"source     : {args.source}")
    print(f"tokenizer  : {args.tokenizer}")

    # ----- Phase 1: walk rollouts -----
    print("\n=== Phase 1: walk benchmark rollouts ===")
    t0 = time.time()
    n_total, n_medical, per_prompt = aggregate_rollouts(args.rollout_dir)
    # `aggregate_rollouts` filters to {qa, conversation} — but benchmark rollouts
    # have eval_mode in {mcq, numeric, medec_hybrid, text_judge_conv, "", None}.
    # We need to re-walk and keep ALL non-empty rollouts because the bench dispatch
    # may emit any of those eval_modes. The helper doesn't take a filter list so
    # we re-implement the loop locally.
    print(f"  (aggregate_rollouts kept qa/conversation only — n={len(per_prompt)};"
          " re-walking to keep all eval_modes)")
    import glob
    by_input: dict[str, dict] = {}
    n_total = 0
    n_kept = 0
    for fp in sorted(glob.glob(os.path.join(args.rollout_dir, "*.jsonl"))):
        with open(fp) as f:
            for line in f:
                n_total += 1
                r = json.loads(line)
                # Bench rollouts may have eval_mode = "" or None on some rows.
                # Take everything — we'll trust the source-side filter (data_source ∈ _BENCH_DATA_SOURCES)
                # to keep us within the benchmark pool.
                inp = r.get("input")
                if not inp:
                    continue
                n_kept += 1
                s = by_input.setdefault(inp, {"scores": [], "em": set(), "steps": set()})
                s["scores"].append(float(r["score"]))
                em = r.get("reward/eval_mode") or ""
                if em:
                    s["em"].add(em)
                # extract step from path: ".../{N}.jsonl"
                try:
                    step = int(os.path.basename(fp).split(".")[0])
                    s["steps"].add(step)
                except Exception:
                    pass
    print(f"  total rollout rows: {n_total}   kept (input present): {n_kept}")
    print(f"  distinct rollout inputs: {len(by_input)}")
    per_prompt = []
    for inp, s in by_input.items():
        per_prompt.append({
            "input":         inp,
            "mean_score":    sum(s["scores"]) / len(s["scores"]),
            "n_samples":     len(s["scores"]),
            "n_steps_seen":  len(s["steps"]),
            "eval_mode":     sorted(s["em"]),
        })
    em_counts = Counter(",".join(p["eval_mode"]) for p in per_prompt)
    for em, c in sorted(em_counts.items()):
        print(f"    eval_mode={em!r}: {c} distinct prompts")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ----- Phase 2: render source (medical_benchmark_* rows only) -----
    print("\n=== Phase 2: render bench rows from source ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.time()
    rendered, ds_kept, bench_idxs = render_bench_rows(tok, args.source)
    print(f"  bench rows in source: {len(rendered)}   rendered in {time.time() - t0:.1f}s")
    print(f"  bench data_source counts: {dict(Counter(ds_kept))}")
    rendered_to_local, collisions = build_rendered_index(rendered)
    # rendered_to_local maps rendered_str → LOCAL index into the bench-only list.
    # We need a map to the GLOBAL source_index so the build script can slice the
    # stage-3 source parquet directly.
    rendered_to_source: dict[str, int] = {
        k: bench_idxs[local] for k, local in rendered_to_local.items()
    }
    print(f"  distinct rendered keys: {len(rendered_to_source)}   collisions (kept first): {collisions}")

    # ----- Phase 3: match rollouts → source rows -----
    print("\n=== Phase 3: match rollouts → bench source rows ===")
    seen_source_indices: set[int] = set()
    orphans = 0
    for p in per_prompt:
        idx = rendered_to_source.get(p["input"])
        if idx is None:
            orphans += 1
            p["source_index"] = None
        else:
            p["source_index"] = idx
            seen_source_indices.add(idx)
    matched = len(per_prompt) - orphans
    print(f"  matched: {matched} / {len(per_prompt)}   orphans={orphans}")
    print(f"  distinct bench source rows seen: {len(seen_source_indices)} / {len(bench_idxs)}")

    # Unseen pool
    bench_all = set(bench_idxs)
    unseen_bench_source_indices = sorted(bench_all - seen_source_indices)
    print(f"  unseen bench source indices: {len(unseen_bench_source_indices)}")
    assert len(seen_source_indices) + len(unseen_bench_source_indices) == len(bench_all), \
        "seen + unseen must sum to total bench rows"

    # ----- Phase 4: threshold what-ifs -----
    print("\n=== Phase 4: threshold what-ifs (mean_score < threshold ⇒ unmastered) ===")
    matched_pp = [p for p in per_prompt if p["source_index"] is not None]
    for thr in thresholds:
        carry = sum(1 for p in matched_pp if p["mean_score"] <  thr)
        mast  = sum(1 for p in matched_pp if p["mean_score"] >= thr)
        print(f"  threshold={thr:>5.2f}:  unmastered={carry:>5}   mastered={mast:>5}")

    if matched_pp:
        scores = [p["mean_score"] for p in matched_pp]
        qs = statistics.quantiles(scores, n=100) if len(scores) >= 100 else None
        print(f"  mean score across seen bench prompts: {statistics.fmean(scores):.4f}")
        if qs:
            print(f"  percentiles: P10={qs[9]:.3f} P50={qs[49]:.3f} P90={qs[89]:.3f}")

    # ----- Phase 5: emit cache -----
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cache = {
        "meta": {
            "rollout_dir": args.rollout_dir,
            "source_parquet": args.source,
            "tokenizer": args.tokenizer,
            "n_rollout_rows": n_total,
            "n_kept_rollout_rows": n_kept,
            "n_distinct_prompts": len(per_prompt),
            "n_orphans": orphans,
            "n_bench_rows_in_source": len(bench_idxs),
            "thresholds_reported": thresholds,
        },
        "per_prompt": [
            {
                "input_hash": _hash16(p["input"]),
                "mean_score": round(p["mean_score"], 4),
                "n_samples": p["n_samples"],
                "n_steps_seen": p["n_steps_seen"],
                "eval_mode": p["eval_mode"],
                "source_index": p["source_index"],
            }
            for p in per_prompt
        ],
        "seen_source_indices": sorted(seen_source_indices),
        "unseen_bench_source_indices": unseen_bench_source_indices,
    }
    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
