"""Assemble the medical_qa STAGE-3 training parquet from the stage-3 cache.

Four pools (CLI knobs select each pool's size):
  A1m  — stage-1 mastered sample        (random sample from rows where stage-1 mean_score ≥ threshold)
  A2u  — stage-2 unmastered sample      (random sample from rows where stage-2 mean_score < threshold)
  A2m  — stage-2 mastered sample        (random sample from rows where stage-2 mean_score ≥ threshold)
  E_*  — benchmarks per-eval_mode sample (from already-deduped buckets in the cache)

All four pools are concatenated, schemas reconciled, and the result is written
as `data/medical_qa_stage3/train.parquet`. Build also writes:
  - `_build_report.json`           — per-pool counts, parameters, totals
  - `seen_prompts_manifest.json`   — running record of seen indices across stages

Reuses helpers from `build_medical_qa_next.py` (pool slicing + struct coercion +
string-to-large_string promotion) and `medical_qa_compat.transform_table`
(benchmarks reward_model.ground_truth struct → JSON string).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from full_mix.preprocess.medical_qa_compat import transform_table  # noqa: E402
from full_mix.curriculum.build_curated_subset import _promote_strings_to_large  # noqa: E402
from full_mix.curriculum.build_medical_qa_next import _coerce_struct_column  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================
def _take(parquet_path: str, indices: list[int]) -> pa.Table:
    t = pq.read_table(parquet_path)
    return t.take(pa.array(indices, type=pa.int64()))


def _sample_n(pool: list[int], n: int, seed: int, label: str) -> list[int]:
    if n >= len(pool):
        if n > len(pool):
            print(f"  WARNING: {label} requested {n} but pool has {len(pool)}; taking all.")
        return sorted(pool)
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


def _split_by_threshold(per_prompt: list[dict], threshold: float) -> tuple[list[int], list[int]]:
    """Return (unmastered_indices, mastered_indices) — sorted, deduped."""
    carry, mast = set(), set()
    for p in per_prompt:
        if p["source_index"] is None:
            continue
        if p["mean_score"] < threshold:
            carry.add(p["source_index"])
        else:
            mast.add(p["source_index"])
    return sorted(carry), sorted(mast)


# ============================================================================
# Schema reconciliation
# ============================================================================
_REWARD_MODEL_CANONICAL_TYPE = pa.struct([
    pa.field("style", pa.string()),
    pa.field("ground_truth", pa.string()),
])


def _build_unified_extra_info_type(stage2_extra_info_type: pa.StructType) -> pa.StructType:
    """Stage-2's extra_info type + 2 new bench-only fields (task_type, helm_scenario).

    Replay rows get None for the two bench fields; bench rows get None for
    fields they don't carry. `_coerce_struct_column`'s pylist round-trip
    fills missing keys with None automatically.
    """
    bench_extras = [
        pa.field("task_type",     pa.string()),
        pa.field("helm_scenario", pa.string()),
    ]
    return pa.struct(list(stage2_extra_info_type) + bench_extras)


def _reconcile(t: pa.Table, unified_ei_type: pa.StructType) -> pa.Table:
    """Coerce extra_info and reward_model to canonical types, then promote
    string columns to large_string."""
    if t.schema.field("extra_info").type != unified_ei_type:
        t = _coerce_struct_column(t, "extra_info", unified_ei_type)
    if t.schema.field("reward_model").type != _REWARD_MODEL_CANONICAL_TYPE:
        t = _coerce_struct_column(t, "reward_model", _REWARD_MODEL_CANONICAL_TYPE)
    promoted = _promote_strings_to_large(t.schema)
    if promoted != t.schema:
        t = t.cast(promoted)
    return t


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache", required=True,
                    help="_stats_cache.json from stats_medical_qa_stage3.py.")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--stage2_unmastered_n", type=int, required=True)
    ap.add_argument("--stage2_mastered_n",   type=int, required=True)
    ap.add_argument("--stage1_mastered_n",   type=int, required=True)
    ap.add_argument("--bench_mcq_n",         type=int, required=True)
    ap.add_argument("--bench_numeric_n",     type=int, required=True)
    ap.add_argument("--bench_medec_n",       type=int, required=True)
    ap.add_argument("--bench_text_n",        type=int, required=True)
    ap.add_argument("--output", required=True,
                    help="Output dir (train.parquet + reports land here).")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print(f"[build_medical_qa_stage3] cache={args.cache}")
    print(f"[build_medical_qa_stage3] threshold={args.threshold}  seed={args.seed}")
    print(f"[build_medical_qa_stage3] output={args.output}")

    with open(args.cache) as f:
        cache = json.load(f)

    s1_src = cache["stage1"]["source_parquet"]
    s2_src = cache["stage2"]["source_parquet"]
    bench_pq = cache["bench"]["bench_parquet"]
    print(f"\nsource parquets:")
    print(f"  stage-1: {s1_src}")
    print(f"  stage-2: {s2_src}")
    print(f"  bench  : {bench_pq}")

    # ----- Phase 1: split per-stage by threshold -----
    s1_carry, s1_mast = _split_by_threshold(cache["stage1"]["per_prompt"], args.threshold)
    s2_carry, s2_mast = _split_by_threshold(cache["stage2"]["per_prompt"], args.threshold)
    print(f"\nstage-1 split at threshold {args.threshold}: carry={len(s1_carry)}  mastered={len(s1_mast)}")
    print(f"stage-2 split at threshold {args.threshold}: carry={len(s2_carry)}  mastered={len(s2_mast)}")

    # ----- Phase 2: sample each replay pool -----
    a1m_idx = _sample_n(s1_mast,  args.stage1_mastered_n,   args.seed + 0, "stage1_mastered")
    a2u_idx = _sample_n(s2_carry, args.stage2_unmastered_n, args.seed + 1, "stage2_unmastered")
    a2m_idx = _sample_n(s2_mast,  args.stage2_mastered_n,   args.seed + 2, "stage2_mastered")
    print(f"\nreplay pool sample sizes:")
    print(f"  A1m (stage-1 mastered):   {len(a1m_idx):>5}")
    print(f"  A2u (stage-2 unmastered): {len(a2u_idx):>5}")
    print(f"  A2m (stage-2 mastered):   {len(a2m_idx):>5}")

    # ----- Phase 3: bench per-bucket sample -----
    bench_buckets = cache["bench"]["buckets_after_dedup"]
    bench_pool_idxs = {
        "mcq":             _sample_n(bench_buckets.get("mcq", []),             args.bench_mcq_n,     args.seed + 3, "bench_mcq"),
        "numeric":         _sample_n(bench_buckets.get("numeric", []),         args.bench_numeric_n, args.seed + 4, "bench_numeric"),
        "medec_hybrid":    _sample_n(bench_buckets.get("medec_hybrid", []),    args.bench_medec_n,   args.seed + 5, "bench_medec"),
        "text_judge_conv": _sample_n(bench_buckets.get("text_judge_conv", []), args.bench_text_n,    args.seed + 6, "bench_text"),
    }
    print(f"\nbench pool sample sizes:")
    for em, idxs in bench_pool_idxs.items():
        print(f"  {em:>18}: {len(idxs):>5}")
    all_bench_idxs = sorted(set().union(*bench_pool_idxs.values()))
    assert len(all_bench_idxs) == sum(len(v) for v in bench_pool_idxs.values()), \
        "bench buckets must be disjoint"

    # ----- Phase 4: slice + transform_table on bench -----
    print("\nslicing pools…")
    pool_a1m = _take(s1_src, a1m_idx)
    pool_a2u = _take(s2_src, a2u_idx)
    pool_a2m = _take(s2_src, a2m_idx)
    pool_bench_raw = _take(bench_pq, all_bench_idxs)
    pool_bench = transform_table(pool_bench_raw)  # struct GT → JSON string
    print(f"  A1m={pool_a1m.num_rows}  A2u={pool_a2u.num_rows}  A2m={pool_a2m.num_rows}  E={pool_bench.num_rows}")

    # ----- Phase 5: reconcile schemas + concat -----
    print("\nreconciling schemas…")
    # The unified extra_info type extends stage-2's. Stage-1's extra_info has
    # the same 17 fields (both flow through the same medical_qa_compat pipe),
    # so using stage-2's is safe.
    s2_ei_type = pool_a2u.schema.field("extra_info").type
    unified_ei_type = _build_unified_extra_info_type(s2_ei_type)
    print(f"  unified extra_info has {len(unified_ei_type)} fields "
          f"(stage-2 base {len(s2_ei_type)} + bench extras task_type/helm_scenario)")

    tables = [
        _reconcile(pool_a1m,   unified_ei_type),
        _reconcile(pool_a2u,   unified_ei_type),
        _reconcile(pool_a2m,   unified_ei_type),
        _reconcile(pool_bench, unified_ei_type),
    ]

    try:
        final = pa.concat_tables(tables, promote_options="default")
    except pa.lib.ArrowInvalid as e:
        print("\nconcat failed; per-pool schemas:")
        for i, t in enumerate(tables):
            print(f"  pool {i} schema:\n{t.schema}")
        raise

    print(f"\nfinal rows: {final.num_rows}")
    print(f"data_source counts: {dict(Counter(final.column('data_source').to_pylist()))}")

    # Sanity: ground_truth must be a string everywhere.
    for r in final.column("reward_model").to_pylist()[:5]:
        assert isinstance(r["ground_truth"], str), \
            f"ground_truth not a string: {type(r['ground_truth']).__name__}"
    print("ground_truth is str for first 5 rows ✓")

    # ----- Phase 6: write outputs -----
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.parquet")
    pq.write_table(final, train_path)
    size_mb = os.path.getsize(train_path) / (1024 * 1024)
    print(f"\nwrote {train_path}  ({size_mb:.1f} MB)")

    report = {
        "cache": args.cache,
        "threshold": args.threshold,
        "seed": args.seed,
        "pool_targets_requested": {
            "stage1_mastered_n":  args.stage1_mastered_n,
            "stage2_unmastered_n": args.stage2_unmastered_n,
            "stage2_mastered_n":  args.stage2_mastered_n,
            "bench_mcq_n":        args.bench_mcq_n,
            "bench_numeric_n":    args.bench_numeric_n,
            "bench_medec_n":      args.bench_medec_n,
            "bench_text_n":       args.bench_text_n,
        },
        "pools_produced": {
            "A1m_stage1_mastered":   {"rows": pool_a1m.num_rows,   "pool_total": len(s1_mast)},
            "A2u_stage2_unmastered": {"rows": pool_a2u.num_rows,   "pool_total": len(s2_carry)},
            "A2m_stage2_mastered":   {"rows": pool_a2m.num_rows,   "pool_total": len(s2_mast)},
            "E_bench":               {"rows": pool_bench.num_rows, "per_eval_mode": {em: len(idxs) for em, idxs in bench_pool_idxs.items()}},
        },
        "output_train_parquet": train_path,
        "total_rows": final.num_rows,
        "total_data_source_counts": dict(Counter(final.column("data_source").to_pylist())),
    }
    with open(os.path.join(args.output, "_build_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {os.path.join(args.output, '_build_report.json')}")

    # Manifest = cumulative seen indices the model has been rolled out on,
    # plus bench rows being introduced this build.
    manifest = {
        "stage1_source": s1_src,
        "stage1_seen_indices": cache["stage1"]["seen_source_indices"],
        "stage2_source": s2_src,
        "stage2_seen_indices": cache["stage2"]["seen_source_indices"],
        "bench_parquet": bench_pq,
        "bench_added_indices": all_bench_idxs,
        "stage_label": "before_stage3",
        "comment": (
            "Cumulative record of medical prompts the model has either been "
            "rolled out on (stage-1 + stage-2 seen) or is being introduced "
            "to in stage-3 (bench_added_indices). The next stage's stats "
            "script should dedupe candidate pools against the union of "
            "these three sets."
        ),
    }
    with open(os.path.join(args.output, "seen_prompts_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {os.path.join(args.output, 'seen_prompts_manifest.json')}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
