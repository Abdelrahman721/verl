"""Layer 6 250 retention prompts into stage-4 train.parquet.

Three pools (per user spec):
  - General (2 000)  — chat / ifeval / safety, sampled proportional to source size
  - Medical (3 000)  — UNSEEN ∪ UNMASTERED prompts vs stage-1 rollouts, threshold 0.6
  - Bench   (1 250)  — UNSEEN ∪ UNMASTERED prompts vs medical_qa_benchmark_rollouts

Outputs:
  - data/medical_qa_stage4/train_coding_only.parquet  — snapshot of the
    pre-retention coding train.parquet (for easy rollback)
  - data/medical_qa_stage4/train.parquet              — REPLACED: now contains
    the 25 000 coding rows + 6 250 retention rows = 31 250 total
  - data/medical_qa_stage4/_retention_report.json     — per-pool counts +
    parameters + the unified extra_info field list

Reuses verbatim from earlier stage-4 build:
  - _coerce_struct_column (build_medical_qa_next.py)
  - _promote_strings_to_large (build_curated_subset.py)
  - _REWARD_MODEL_CANONICAL_TYPE / _drop_non_canonical_columns (stage-4 build)

Usage:
    python -m full_mix.curriculum.build_medical_qa_stage4_retention \\
        --existing_train data/medical_qa_stage4/train.parquet \\
        --medical_cache  data/medical_qa_stage3/_stats_cache.json \\
        --bench_cache    data/medical_qa_stage4/_bench_retention_cache.json \\
        --general_dir    data/full_mix/train \\
        --medical_source data/medical_qa/train.parquet \\
        --bench_source   data/medical_qa_stage3/train.parquet \\
        --output_dir     data/medical_qa_stage4/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from full_mix.curriculum.build_curated_subset import _promote_strings_to_large  # noqa: E402
from full_mix.curriculum.build_medical_qa_next import _coerce_struct_column  # noqa: E402
from full_mix.curriculum.build_medical_qa_stage4 import (  # noqa: E402
    _REWARD_MODEL_CANONICAL_TYPE,
    _drop_non_canonical_columns,
)


# ============================================================================
# Pool 1 — General retention
# ============================================================================
_GENERAL_FILES = ("chat_with_baseline_train.parquet",
                  "ifeval_train.parquet",
                  "safety_train.parquet")


def _sample_general(general_dir: str, budget: int, seed: int) -> dict[str, list[int]]:
    """Sample indices from each of the three general-retention parquets,
    proportional to source size, total = `budget`."""
    sizes = {}
    for fn in _GENERAL_FILES:
        meta = pq.read_metadata(os.path.join(general_dir, fn))
        sizes[fn] = meta.num_rows
    total = sum(sizes.values())
    # Largest-remainder rounding so the budget is hit exactly.
    raw = {fn: budget * n / total for fn, n in sizes.items()}
    floor = {fn: int(v) for fn, v in raw.items()}
    remainder = budget - sum(floor.values())
    # Give the extra slots to the largest fractional parts.
    order = sorted(raw.items(), key=lambda x: -(x[1] - int(x[1])))
    for fn, _ in order[:remainder]:
        floor[fn] += 1
    print(f"  general source sizes: {sizes}  → counts: {floor}")
    assert sum(floor.values()) == budget, "general split must equal budget"

    rng = random.Random(seed)
    out: dict[str, list[int]] = {}
    for fn, n in floor.items():
        idxs = rng.sample(range(sizes[fn]), n) if n < sizes[fn] else list(range(sizes[fn]))
        out[fn] = sorted(idxs)
    return out


# ============================================================================
# Pool 2 — Medical retention (from existing stage-3 cache)
# ============================================================================
def _build_medical_pool(medical_cache: dict, source_row_count: int,
                       threshold: float, mastered_budget: int,
                       unseen_budget: int, seed: int) -> list[int]:
    """Sample MASTERED + UNSEEN stage-1 prompts as the medical retention pool.

    `mastered_budget` rows are drawn from prompts seen in rollouts AND scored
    above threshold (retention proper). `unseen_budget` rows are drawn from
    prompts never sampled at all (new exposure). Uniform random within each
    bucket; the two buckets are joined and returned as sorted indices.

    Previous version sampled uniformly from UNSEEN ∪ UNMASTERED, which made
    the seen-bucket vanish (≈61/3000 with the default proportions). The user
    asked for an explicit oversample of the mastered bucket.
    """
    stage1 = medical_cache["stage1"]
    seen = set(stage1["seen_source_indices"])
    mastered_set = {p["source_index"] for p in stage1["per_prompt"]
                    if p["source_index"] is not None and p["mean_score"] >= threshold}
    unmastered_set = {p["source_index"] for p in stage1["per_prompt"]
                      if p["source_index"] is not None and p["mean_score"] < threshold}
    unseen_set = set(range(source_row_count)) - seen

    print(f"  stage-1 seen: {len(seen)}  unseen: {len(unseen_set)}")
    print(f"  seen split at threshold {threshold}: mastered={len(mastered_set)}  "
          f"unmastered={len(unmastered_set)}")
    print(f"  budget: mastered={mastered_budget}  unseen={unseen_budget}  "
          f"(total={mastered_budget + unseen_budget})")

    rng = random.Random(seed)
    mastered_idxs = (sorted(mastered_set) if mastered_budget >= len(mastered_set)
                     else rng.sample(sorted(mastered_set), mastered_budget))
    unseen_idxs   = (sorted(unseen_set)   if unseen_budget   >= len(unseen_set)
                     else rng.sample(sorted(unseen_set),   unseen_budget))
    if mastered_budget > len(mastered_set):
        print(f"  WARNING: mastered_budget {mastered_budget} > available {len(mastered_set)}; "
              f"taking all available.")
    if unseen_budget > len(unseen_set):
        print(f"  WARNING: unseen_budget {unseen_budget} > available {len(unseen_set)}; "
              f"taking all available.")
    return sorted(set(mastered_idxs) | set(unseen_idxs))


# ============================================================================
# Pool 3 — Bench retention (from new bench-retention cache)
# ============================================================================
def _build_bench_pool(bench_cache: dict, threshold: float, budget: int,
                     seed: int) -> list[int]:
    """Pool = UNSEEN ∪ UNMASTERED bench source indices, sample `budget`."""
    unmastered = {p["source_index"] for p in bench_cache["per_prompt"]
                  if p["source_index"] is not None and p["mean_score"] < threshold}
    unseen = set(bench_cache["unseen_bench_source_indices"])
    pool = sorted(unseen | unmastered)
    print(f"  bench unmastered (<{threshold}): {len(unmastered)}  "
          f"unseen: {len(unseen)}  pool size: {len(pool)}")
    if budget >= len(pool):
        return pool
    rng = random.Random(seed)
    return sorted(rng.sample(pool, budget))


# ============================================================================
# Schema reconciliation
# ============================================================================
def _build_super_extra_info_type(*types: pa.StructType) -> pa.StructType:
    """Union the field name sets across N struct types. The first occurrence
    of each name wins (its type is kept). Always ensures `task_type` exists."""
    seen: dict[str, pa.Field] = {}
    for t in types:
        for i in range(t.num_fields):
            f = t.field(i)
            if f.name not in seen:
                seen[f.name] = f
    fields = list(seen.values())
    if "task_type" not in seen:
        fields.append(pa.field("task_type", pa.string()))
    return pa.struct(fields)


def _normalise_reward_model(t: pa.Table) -> pa.Table:
    """Force the canonical {style: string, ground_truth: string} type.

    Every retention source already has string ground_truth — we just need to
    ensure the struct field order/type matches the stage-4 canonical schema
    so concat doesn't complain.
    """
    records = t.column("reward_model").to_pylist()
    new_rm = []
    for r in records:
        if not isinstance(r, dict):
            new_rm.append({"style": "", "ground_truth": ""})
            continue
        gt = r.get("ground_truth")
        if isinstance(gt, str):
            gt_str = gt
        elif gt is None:
            gt_str = ""
        else:
            gt_str = json.dumps(gt, ensure_ascii=False, default=str)
        new_rm.append({"style": r.get("style", "") or "", "ground_truth": gt_str})
    new_arr = pa.array(new_rm, type=_REWARD_MODEL_CANONICAL_TYPE)
    idx = t.column_names.index("reward_model")
    cols = list(t.columns)
    cols[idx] = new_arr
    return pa.table(cols, names=t.column_names)


def _reconcile(t: pa.Table, unified_ei_type: pa.StructType) -> pa.Table:
    """Coerce extra_info + reward_model + promote strings."""
    if t.schema.field("extra_info").type != unified_ei_type:
        t = _coerce_struct_column(t, "extra_info", unified_ei_type)
    if t.schema.field("reward_model").type != _REWARD_MODEL_CANONICAL_TYPE:
        t = _normalise_reward_model(t)
    promoted = _promote_strings_to_large(t.schema)
    if promoted != t.schema:
        t = t.cast(promoted)
    return t


# ============================================================================
# Misc
# ============================================================================
def _take(parquet_path: str, indices: list[int]) -> pa.Table:
    t = pq.read_table(parquet_path)
    return t.take(pa.array(indices, type=pa.int64()))


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--existing_train", required=True,
                    help="Current coding-only stage-4 train.parquet.")
    ap.add_argument("--medical_cache", required=True,
                    help="data/medical_qa_stage3/_stats_cache.json (stage1 block is consumed).")
    ap.add_argument("--bench_cache", required=True,
                    help="Output of stats_medical_qa_bench_retention.py.")
    ap.add_argument("--general_dir", required=True,
                    help="Dir with chat_with_baseline_train.parquet / ifeval_train.parquet / safety_train.parquet.")
    ap.add_argument("--medical_source", required=True,
                    help="data/medical_qa/train.parquet (stage-1 source).")
    ap.add_argument("--bench_source", required=True,
                    help="data/medical_qa_stage3/train.parquet (the bench-row source).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--threshold",       type=float, default=0.6)
    ap.add_argument("--general_budget",  type=int,   default=2000)
    # Medical: split explicitly into mastered + unseen (no longer uniform from union)
    ap.add_argument("--medical_mastered_budget", type=int, default=400,
                    help="rows drawn from stage-1 mastered (mean_score ≥ threshold)")
    ap.add_argument("--medical_unseen_budget",   type=int, default=2600,
                    help="rows drawn from stage-1 unseen (never in rollouts)")
    ap.add_argument("--bench_budget",    type=int,   default=1250)
    ap.add_argument("--seed",            type=int,   default=1234)
    args = ap.parse_args()

    print(f"[build_medical_qa_stage4_retention]")
    print(f"  threshold                = {args.threshold}")
    print(f"  general_budget           = {args.general_budget}")
    print(f"  medical_mastered_budget  = {args.medical_mastered_budget}")
    print(f"  medical_unseen_budget    = {args.medical_unseen_budget}")
    print(f"  bench_budget             = {args.bench_budget}")
    print(f"  seed                     = {args.seed}")
    print()

    # ----- Load caches -----
    with open(args.medical_cache) as f:
        medical_cache = json.load(f)
    with open(args.bench_cache) as f:
        bench_cache = json.load(f)
    medical_source_rows = medical_cache["stage1"]["src_total_rows"]
    bench_n_in_source   = bench_cache["meta"]["n_bench_rows_in_source"]
    print(f"  medical source rows: {medical_source_rows}   "
          f"bench rows in stage-3 source: {bench_n_in_source}")

    # ----- Phase 1: snapshot existing train -----
    print("\n=== Phase 1: snapshot existing coding-only train ===")
    snapshot = os.path.join(args.output_dir, "train_coding_only.parquet")
    if not os.path.exists(snapshot):
        shutil.copy2(args.existing_train, snapshot)
        print(f"  saved {snapshot}")
    else:
        print(f"  snapshot already exists at {snapshot} (not overwriting)")

    # ----- Phase 2: sample each pool -----
    print("\n=== Phase 2: General pool ===")
    general_idxs = _sample_general(args.general_dir, args.general_budget, args.seed)

    print("\n=== Phase 2: Medical pool ===")
    medical_idxs = _build_medical_pool(medical_cache, medical_source_rows,
                                       args.threshold,
                                       args.medical_mastered_budget,
                                       args.medical_unseen_budget,
                                       args.seed + 1)
    print(f"  sampled medical indices: {len(medical_idxs)}")

    print("\n=== Phase 2: Bench pool ===")
    bench_idxs = _build_bench_pool(bench_cache, args.threshold, args.bench_budget,
                                   args.seed + 2)
    print(f"  sampled bench indices: {len(bench_idxs)}")

    # ----- Phase 3: slice each pool -----
    print("\n=== Phase 3: slice each pool ===")
    pool_tables: list[tuple[str, pa.Table]] = []

    # General — three sub-pools
    for fn, idxs in general_idxs.items():
        path = os.path.join(args.general_dir, fn)
        t = _take(path, idxs)
        t = _drop_non_canonical_columns(t)
        pool_tables.append((f"general/{fn}", t))
        print(f"  general/{fn}: {t.num_rows} rows")

    # Medical
    med_t = _take(args.medical_source, medical_idxs)
    med_t = _drop_non_canonical_columns(med_t)
    pool_tables.append(("medical", med_t))
    print(f"  medical: {med_t.num_rows} rows  "
          f"data_source: {dict(Counter(med_t.column('data_source').to_pylist()))}")

    # Bench
    bench_t = _take(args.bench_source, bench_idxs)
    bench_t = _drop_non_canonical_columns(bench_t)
    pool_tables.append(("bench", bench_t))
    print(f"  bench: {bench_t.num_rows} rows  "
          f"data_source: {dict(Counter(bench_t.column('data_source').to_pylist()))}")

    # ----- Phase 4: schema reconciliation -----
    print("\n=== Phase 4: schema reconciliation ===")
    existing = pq.read_table(args.existing_train)
    existing_ei = existing.schema.field("extra_info").type
    print(f"  existing stage-4 train: {existing.num_rows} rows  "
          f"extra_info fields: {existing_ei.num_fields}")

    # Build super-union extra_info type from existing + all pools.
    all_ei_types = [existing_ei] + [t.schema.field("extra_info").type for _, t in pool_tables]
    unified_ei = _build_super_extra_info_type(*all_ei_types)
    print(f"  unified extra_info: {unified_ei.num_fields} fields  "
          f"(={[f.name for f in unified_ei]})")

    # Reconcile every pool AND the existing train (so its extra_info widens to the union too).
    reconciled = [_reconcile(existing, unified_ei)]
    for label, t in pool_tables:
        r = _reconcile(t, unified_ei)
        reconciled.append(r)
        print(f"  reconciled {label}: {r.num_rows} rows")

    # ----- Phase 5: concat -----
    print("\n=== Phase 5: concat ===")
    final = pa.concat_tables(reconciled, promote_options="default")
    print(f"  total rows: {final.num_rows}")
    print(f"  data_source counts: {dict(Counter(final.column('data_source').to_pylist()))}")
    tt_counts = Counter(
        (ei or {}).get("task_type") for ei in final.column("extra_info").to_pylist()
    )
    print(f"  task_type counts (None = no bench routing): {dict(tt_counts)}")

    # Sanity: ground_truth is str everywhere
    for r in final.column("reward_model").to_pylist()[:5]:
        assert isinstance(r["ground_truth"], str), \
            f"ground_truth not str: {type(r['ground_truth']).__name__}"
    print("  ground_truth is str for first 5 rows ✓")

    # ----- Phase 6: replace train.parquet + write report -----
    print("\n=== Phase 6: write outputs ===")
    train_path = os.path.join(args.output_dir, "train.parquet")
    pq.write_table(final, train_path)
    size_mb = os.path.getsize(train_path) / (1024 * 1024)
    print(f"  wrote {train_path}  ({size_mb:.1f} MB)")

    report = {
        "params": {
            "threshold":                args.threshold,
            "general_budget":           args.general_budget,
            "medical_mastered_budget":  args.medical_mastered_budget,
            "medical_unseen_budget":    args.medical_unseen_budget,
            "bench_budget":             args.bench_budget,
            "seed":                     args.seed,
        },
        "pools": {
            "general_per_source":            {fn: len(v) for fn, v in general_idxs.items()},
            "general_total":                 args.general_budget,
            "medical_sampled":               len(medical_idxs),
            "medical_mastered_requested":    args.medical_mastered_budget,
            "medical_unseen_requested":      args.medical_unseen_budget,
            "bench_sampled":                 len(bench_idxs),
            "existing_train_pre_retention":  existing.num_rows,
            "final_rows":                    final.num_rows,
        },
        "data_source_counts": dict(Counter(final.column("data_source").to_pylist())),
        "task_type_counts":   {str(k): v for k, v in tt_counts.items()},
        "unified_extra_info_fields": [f.name for f in unified_ei],
        "snapshot_before_retention": snapshot,
        "output_train_parquet":      train_path,
    }
    report_path = os.path.join(args.output_dir, "_retention_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  wrote {report_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
