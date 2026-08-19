"""Stats for assembling the medical_qa STAGE-3 training file.

Generalises `stats_medical_qa_next.py` for the two-rollout-dir + benchmarks
case. The build script (`build_medical_qa_stage3.py`) consumes the cache this
emits.

Three input streams are analysed:
  1. Stage-1 rollouts (`medical-qa-fresh`) against `data/medical_qa/train.parquet`
  2. Stage-2 rollouts (`medical-qa-stage2`) against `data/medical_qa_stage2/train.parquet`
  3. The new benchmarks parquet — bucketed by `extra_info.helm_scenario` /
     `extra_info.task_type` using the same routing tables as the new reward
     module (`tmp.py` → soon `qa_openrouter_bench.py`).

Why a single combined cache:
  - The build script needs per-prompt scores for BOTH stages to apply one
    threshold and produce three replay pools (stage-2 unmastered, stage-2
    mastered, stage-1 mastered).
  - The benchmarks pool must dedupe against the UNION seen-set, which lives
    on the chat-template-rendered prompt string. Doing this in one pass
    avoids re-walking 800k+ JSONL rows during build.

Usage:
    python -m full_mix.curriculum.stats_medical_qa_stage3 \\
        --stage1_rollout_dir /data/abdelrahman/verl/medical_qa_rollouts \\
        --stage1_source      /data/abdelrahman/verl/data/medical_qa/train.parquet \\
        --stage2_rollout_dir /data/abdelrahman/verl/medical_qa_stage2_rollouts \\
        --stage2_source      /data/abdelrahman/verl/data/medical_qa_stage2/train.parquet \\
        --tokenizer          /data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh/global_step_500/merged_hf_model \\
        --bench              /data/hazem/medical-data-gen/processed/rl/benchmarks/v1/train.parquet \\
        --out                /data/abdelrahman/verl/data/medical_qa_stage3/_stats_cache.json
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


# ============================================================================
# Routing tables — MUST stay in lockstep with qa_openrouter_bench.py (currently
# living at /data/abdelrahman/verl/tmp.py). If the source there changes, mirror
# it here so the stats-time bucket counts match what the trainer will see.
# ============================================================================
_HELM_SCENARIO_TO_EVAL_MODE: dict[str, str] = {
    "medcalc_bench":  "numeric",
    "medec":          "medec_hybrid",
    "aci_bench":      "text_judge_conv",
    "dischargeme":    "text_judge_conv",
    "med_dialog":     "text_judge_conv",
    "medi_qa":        "text_judge_conv",
    "mimic_bhc":      "text_judge_conv",
}
_TASK_TYPE_TO_EVAL_MODE: dict[str, str] = {
    "mcq":            "mcq",
    "classification": "mcq",
}


def bench_eval_mode(extra_info) -> str | None:
    """Mirror of `_bench_determine_eval_mode` from qa_openrouter_bench.py."""
    if not isinstance(extra_info, dict):
        return None
    hs = extra_info.get("helm_scenario")
    if hs and hs in _HELM_SCENARIO_TO_EVAL_MODE:
        return _HELM_SCENARIO_TO_EVAL_MODE[hs]
    tt = extra_info.get("task_type")
    if tt and tt in _TASK_TYPE_TO_EVAL_MODE:
        return _TASK_TYPE_TO_EVAL_MODE[tt]
    return None


# ============================================================================
# Reuse: the chat-template render + rollout walk pipeline already lives in
# stats_medical_qa_next.py. Don't duplicate.
# ============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from full_mix.curriculum.stats_medical_qa_next import (  # noqa: E402
    aggregate_rollouts,
    build_rendered_index,
    render_chat,
    render_parquet_prompts,
)


def _hash16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# Per-stage walker — encapsulates the load+match+score pipeline so we run it
# twice (once per stage) with consistent reporting.
# ============================================================================
def walk_one_stage(label: str, tok, rollout_dir: str, source_parquet: str) -> dict:
    """Walk rollouts, render source, match, return a per-stage summary dict.

    Output dict (also serialised into the cache):
        {
          "label": ...,
          "rollout_dir": ...,
          "source_parquet": ...,
          "n_rollout_rows": int,
          "n_medical_rollout_rows": int,
          "n_distinct_prompts": int,
          "n_orphans": int,
          "src_total_rows": int,
          "src_distinct_rendered": int,
          "src_data_source_counts": {ds: int},
          "per_prompt": [
            {input_hash, mean_score, n_samples, n_steps_seen, eval_mode, source_index},
            ...
          ],
          "seen_source_indices": [int, ...],
        }
    """
    print(f"\n========== {label} ==========")
    print(f"  rollout_dir: {rollout_dir}")
    print(f"  source     : {source_parquet}")

    t0 = time.time()
    n_total, n_medical, per_prompt = aggregate_rollouts(rollout_dir)
    print(f"  walk: {time.time() - t0:.1f}s  rows={n_total}  medical={n_medical}  "
          f"distinct_prompts={len(per_prompt)}")

    t1 = time.time()
    rendered, data_sources = render_parquet_prompts(tok, source_parquet)
    src_idx, collisions = build_rendered_index(rendered)
    print(f"  render: {time.time() - t1:.1f}s  src_rows={len(rendered)}  "
          f"distinct={len(src_idx)}  collisions={collisions}")
    print(f"    data_source counts: {dict(Counter(data_sources))}")

    # Match each distinct rollout prompt back to its source row index.
    seen: set[int] = set()
    orphans = 0
    for p in per_prompt:
        idx = src_idx.get(p["input"])
        if idx is None:
            orphans += 1
            p["source_index"] = None
        else:
            p["source_index"] = idx
            seen.add(idx)
    matched = len(per_prompt) - orphans
    print(f"  matched: {matched}/{len(per_prompt)}  orphans={orphans}  "
          f"distinct_source_rows_seen={len(seen)}")

    # Per-eval-mode score summary (just for the printed report).
    by_em = defaultdict(list)
    for p in per_prompt:
        if p["source_index"] is None:
            continue
        by_em[",".join(p["eval_mode"])].append(p["mean_score"])
    print("\n  per-eval_mode score summary:")
    for em, vals in by_em.items():
        if not vals:
            continue
        mean_v = statistics.fmean(vals)
        if len(vals) >= 100:
            qs = statistics.quantiles(vals, n=100)
            line = (f"    {em!r}: n={len(vals)}  mean={mean_v:.3f}  "
                    f"P10={qs[9]:.3f} P50={qs[49]:.3f} P90={qs[89]:.3f}")
        else:
            line = f"    {em!r}: n={len(vals)}  mean={mean_v:.3f}"
        print(line)

    return {
        "label": label,
        "rollout_dir": rollout_dir,
        "source_parquet": source_parquet,
        "n_rollout_rows": n_total,
        "n_medical_rollout_rows": n_medical,
        "n_distinct_prompts": len(per_prompt),
        "n_orphans": orphans,
        "src_total_rows": len(rendered),
        "src_distinct_rendered": len(src_idx),
        "src_data_source_counts": dict(Counter(data_sources)),
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
        "seen_source_indices": sorted(seen),
        # The rendered strings of the prompts seen in rollouts — used in the
        # bench-dedup step. Not persisted to JSON (large); kept in the
        # returned dict for in-memory use only.
        "_seen_rendered": {p["input"] for p in per_prompt if p["source_index"] is not None},
    }


# ============================================================================
# Benchmarks dedup + bucketing
# ============================================================================
def analyse_bench(tok, bench_parquet: str, seen_rendered_union: set[str]) -> dict:
    """Render the bench parquet, dedup against seen-union, bucket by routing."""
    import pyarrow.parquet as pq

    print("\n========== BENCHMARKS ==========")
    print(f"  parquet: {bench_parquet}")
    t0 = time.time()
    t = pq.read_table(bench_parquet, columns=["prompt", "data_source", "extra_info"])
    prompts = t.column("prompt").to_pylist()
    data_sources = t.column("data_source").to_pylist()
    extra_infos = t.column("extra_info").to_pylist()
    rendered = [render_chat(tok, p) for p in prompts]
    print(f"  render: {time.time() - t0:.1f}s  rows={len(rendered)}")

    # Dedup
    survive_idxs = [i for i, r in enumerate(rendered) if r not in seen_rendered_union]
    print(f"  survives dedup vs seen-union: {len(survive_idxs)} / {len(rendered)}")

    # Bucket by routed eval_mode
    buckets: dict[str, list[int]] = defaultdict(list)
    helm_breakdown: dict[str, Counter] = defaultdict(Counter)
    task_type_breakdown: dict[str, Counter] = defaultdict(Counter)
    fallthrough: list[int] = []
    for i in survive_idxs:
        ei = extra_infos[i] or {}
        em = bench_eval_mode(ei)
        if em is None:
            fallthrough.append(i)
            continue
        buckets[em].append(i)
        if isinstance(ei, dict):
            helm_breakdown[em][ei.get("helm_scenario") or ""] += 1
            task_type_breakdown[em][ei.get("task_type") or ""] += 1
    print("\n  routing buckets (after dedup):")
    for em in ("mcq", "numeric", "medec_hybrid", "text_judge_conv"):
        idxs = buckets.get(em, [])
        print(f"    {em:>18}: {len(idxs):>6}")
        if idxs:
            hs = helm_breakdown[em]
            tt = task_type_breakdown[em]
            for k, c in sorted(hs.items(), key=lambda x: -x[1])[:5]:
                print(f"        helm_scenario={k!r}: {c}")
            for k, c in sorted(tt.items(), key=lambda x: -x[1])[:3]:
                print(f"        task_type    ={k!r}: {c}")
    if fallthrough:
        print(f"  WARN: {len(fallthrough)} survivor rows did not route to any "
              "benchmark eval_mode — they would fall through to plain 'qa'.")
        # Show a tiny sample of their extra_info to help diagnose
        for i in fallthrough[:3]:
            print(f"    e.g. extra_info subset: "
                  f"task_type={extra_infos[i].get('task_type')!r}  "
                  f"helm_scenario={extra_infos[i].get('helm_scenario')!r}")
    return {
        "bench_parquet": bench_parquet,
        "n_total_rows": len(rendered),
        "n_after_dedup": len(survive_idxs),
        "n_fallthrough": len(fallthrough),
        "buckets_after_dedup": {em: idxs for em, idxs in buckets.items()},
        "fallthrough_row_indices": fallthrough,
        "data_source_counts": dict(Counter(data_sources)),
    }


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stage1_rollout_dir", required=True)
    ap.add_argument("--stage1_source", required=True)
    ap.add_argument("--stage2_rollout_dir", required=True)
    ap.add_argument("--stage2_source", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--bench", required=True,
                    help="Path to the benchmarks parquet (renders + dedup + bucketing).")
    ap.add_argument("--out", required=True,
                    help="Output JSON cache path (build script consumes this).")
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8",
                    help="Comma-separated success-rate thresholds for what-if counts.")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    from transformers import AutoTokenizer
    print(f"loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # ----- Per-stage walk -----
    stage1 = walk_one_stage("STAGE-1 (medical-qa-fresh)", tok,
                            args.stage1_rollout_dir, args.stage1_source)
    stage2 = walk_one_stage("STAGE-2 (medical-qa-stage2)", tok,
                            args.stage2_rollout_dir, args.stage2_source)

    # ----- Threshold what-ifs (per stage) -----
    for stage in (stage1, stage2):
        print(f"\n  threshold what-ifs for {stage['label']}:")
        for thr in thresholds:
            carry = sum(1 for p in stage["per_prompt"]
                        if p["source_index"] is not None and p["mean_score"] <  thr)
            mast  = sum(1 for p in stage["per_prompt"]
                        if p["source_index"] is not None and p["mean_score"] >= thr)
            print(f"    threshold={thr:>5.2f}: unmastered={carry:>6}  mastered={mast:>6}")

    # ----- Benchmarks: dedup against union, bucket -----
    seen_union = stage1["_seen_rendered"] | stage2["_seen_rendered"]
    print(f"\nseen-union size (stage-1 ∪ stage-2): {len(seen_union)}")
    bench = analyse_bench(tok, args.bench, seen_union)

    # ----- Emit cache (drop the heavy `_seen_rendered` sets before serialising) -----
    cache = {
        "meta": {
            "tokenizer": args.tokenizer,
            "bench_parquet": args.bench,
            "thresholds_reported": thresholds,
        },
        "stage1": {k: v for k, v in stage1.items() if not k.startswith("_")},
        "stage2": {k: v for k, v in stage2.items() if not k.startswith("_")},
        "bench":  bench,
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out) / (1024*1024):.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
