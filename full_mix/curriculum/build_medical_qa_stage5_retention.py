"""Build the stage-5 retention pool (target 9 000 rows).

Four pools per user spec, sampled independently with offsets from a single
base seed (default 5051):

  1. General      (2 000)  — random from
                              data/full_mix/train/{chat_with_baseline,ifeval,safety}_train.parquet,
                              proportional to source size. Seed = base.
  2. Medical      (3 000)  — UNSEEN-only vs stage-1 rollouts
                              (data/medical_qa_stage3/_stats_cache.json#stage1).
                              Seed = base + 1.
  3. Bench        (1 250)  — UNSEEN in BOTH:
                              - data/medical_qa_stage4/_bench_retention_cache.json (bench rollouts)
                              - data/medical_qa_stage5/_stage4_retention_cache.json (stage-4 coding rollouts)
                              Seed = base + 2.
  4. Coding-unmastered (ALL) — from stage-4 cache, mean_score < threshold,
                              indices into
                              data/medical_qa_stage4/train_coding_only.parquet.
                              (No seed — deterministic take-all.)

If pools 1-4 sum to < 9 000, top up from medical UNSEEN (excluding the
indices already drawn by pool 2). Seed = base + 3.

Output:
  data/medical_qa_stage5/retention.parquet
  data/medical_qa_stage5/_retention_report.json

Schema-reconciliation reuses verbatim:
  - _drop_non_canonical_columns / _REWARD_MODEL_CANONICAL_TYPE     (stage-4 build)
  - _coerce_struct_column                                          (build_medical_qa_next)
  - _promote_strings_to_large                                      (build_curated_subset)
  - _build_super_extra_info_type / _normalise_reward_model         (stage-4 retention)

The output schema is widened to the union of every pool's extra_info plus
`task_type`. When the user wants to fold this into the stage-5 train, the
two parquets are concat'd with the same `_coerce_struct_column` step over
their super-union — that's the same path the stage-4 retention build took.

Usage:
    python -m full_mix.curriculum.build_medical_qa_stage5_retention \\
        --output_dir       /data/abdelrahman/verl/data/medical_qa_stage5/
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

from full_mix.curriculum.build_curated_subset import _promote_strings_to_large  # noqa: E402
from full_mix.curriculum.build_medical_qa_next import _coerce_struct_column  # noqa: E402
from full_mix.curriculum.build_medical_qa_stage4 import (  # noqa: E402
    _REWARD_MODEL_CANONICAL_TYPE,
    _drop_non_canonical_columns,
)
from full_mix.curriculum.build_medical_qa_stage4_retention import (  # noqa: E402
    _build_super_extra_info_type,
    _normalise_reward_model,
)


_GENERAL_FILES = ("chat_with_baseline_train.parquet",
                  "ifeval_train.parquet",
                  "safety_train.parquet")


# ============================================================================
# Pool 1 — General (proportional random)
# ============================================================================
def _sample_general(general_dir: str, budget: int, seed: int) -> dict[str, list[int]]:
    sizes = {fn: pq.read_metadata(os.path.join(general_dir, fn)).num_rows
             for fn in _GENERAL_FILES}
    total = sum(sizes.values())
    raw = {fn: budget * n / total for fn, n in sizes.items()}
    floor = {fn: int(v) for fn, v in raw.items()}
    remainder = budget - sum(floor.values())
    for fn, _ in sorted(raw.items(), key=lambda x: -(x[1] - int(x[1])))[:remainder]:
        floor[fn] += 1
    print(f"  general sizes={sizes}  → counts={floor}")
    assert sum(floor.values()) == budget

    rng = random.Random(seed)
    return {
        fn: sorted(rng.sample(range(sizes[fn]), n) if n < sizes[fn] else list(range(sizes[fn])))
        for fn, n in floor.items()
    }


# ============================================================================
# Pool 2 — Medical UNSEEN (vs stage-1 rollouts)
# ============================================================================
def _medical_unseen_indices(medical_cache: dict) -> list[int]:
    stage1 = medical_cache["stage1"]
    seen = set(stage1["seen_source_indices"])
    total = stage1["src_total_rows"]
    unseen = sorted(set(range(total)) - seen)
    print(f"  stage-1 source rows: {total}  seen: {len(seen)}  unseen: {len(unseen)}")
    return unseen


def _sample_medical(medical_cache: dict, budget: int, seed: int) -> list[int]:
    unseen = _medical_unseen_indices(medical_cache)
    if budget >= len(unseen):
        print(f"  WARNING: budget {budget} >= unseen pool {len(unseen)}; taking all")
        return unseen
    rng = random.Random(seed)
    return sorted(rng.sample(unseen, budget))


# ============================================================================
# Pool 3 — Bench UNSEEN (vs bench rollouts AND vs stage-4 coding rollouts)
# ============================================================================
def _sample_bench(bench_cache: dict, stage4_cache: dict,
                  budget: int, seed: int) -> list[int]:
    unseen_in_bench = set(bench_cache["unseen_bench_source_indices"])
    seen_in_stage4  = set(stage4_cache["seen_bench_source_indices_any"])
    # Final pool = unseen in bench rollouts AND not seen in stage-4 rollouts.
    pool = sorted(unseen_in_bench - seen_in_stage4)
    print(f"  bench unseen-in-bench-rollouts: {len(unseen_in_bench)}")
    print(f"  bench seen-in-stage4-rollouts:  {len(seen_in_stage4)}")
    print(f"  intersection (unseen in BOTH):  {len(pool)}")
    if budget >= len(pool):
        print(f"  WARNING: budget {budget} >= pool size {len(pool)}; taking all")
        return pool
    rng = random.Random(seed)
    return sorted(rng.sample(pool, budget))


# ============================================================================
# Pool 4 — Stage-4 coding UNMASTERED (no budget, take all)
# ============================================================================
def _coding_unmastered_indices(stage4_cache: dict, threshold: float) -> list[int]:
    _CODING_EMS = frozenset({"icd_multi_label", "snomed_single_label"})
    idxs = sorted({
        p["coding_source_index"] for p in stage4_cache["per_prompt"]
        if p["coding_source_index"] is not None
        and p["mean_score"] < threshold
        and p["eval_mode"] and all(em in _CODING_EMS for em in p["eval_mode"])
    })
    print(f"  stage-4 coding-unmastered (mean_score < {threshold}): {len(idxs)} rows")
    return idxs


# ============================================================================
# Top-up — more medical UNSEEN
# ============================================================================
def _topup_medical(medical_cache: dict, exclude: set[int], needed: int,
                   seed: int) -> list[int]:
    if needed <= 0:
        return []
    unseen = _medical_unseen_indices(medical_cache)
    pool = sorted(set(unseen) - exclude)
    print(f"  top-up: need={needed}  pool size={len(pool)} (after excluding pool-2)")
    if needed >= len(pool):
        print(f"  WARNING: top-up needed {needed} >= pool size {len(pool)}; taking all")
        return pool
    rng = random.Random(seed)
    return sorted(rng.sample(pool, needed))


# ============================================================================
# Helpers
# ============================================================================
def _take(parquet_path: str, indices: list[int]) -> pa.Table:
    t = pq.read_table(parquet_path)
    return t.take(pa.array(indices, type=pa.int64()))


def _reconcile(t: pa.Table, unified_ei: pa.StructType) -> pa.Table:
    if t.schema.field("extra_info").type != unified_ei:
        t = _coerce_struct_column(t, "extra_info", unified_ei)
    if t.schema.field("reward_model").type != _REWARD_MODEL_CANONICAL_TYPE:
        t = _normalise_reward_model(t)
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
    ap.add_argument("--output_dir", required=True)
    # Source pointers — sensible defaults, can be overridden.
    ap.add_argument("--general_dir",
                    default="/data/abdelrahman/verl/data/full_mix/train")
    ap.add_argument("--medical_source",
                    default="/data/abdelrahman/verl/data/medical_qa/train.parquet")
    ap.add_argument("--bench_source",
                    default="/data/abdelrahman/verl/data/medical_qa_stage3/train.parquet")
    ap.add_argument("--coding_source",
                    default="/data/abdelrahman/verl/data/medical_qa_stage4/train_coding_only.parquet")
    ap.add_argument("--medical_cache",
                    default="/data/abdelrahman/verl/data/medical_qa_stage3/_stats_cache.json")
    ap.add_argument("--bench_cache",
                    default="/data/abdelrahman/verl/data/medical_qa_stage4/_bench_retention_cache.json")
    ap.add_argument("--stage4_cache",
                    default="/data/abdelrahman/verl/data/medical_qa_stage5/_stage4_retention_cache.json")
    # Knobs.
    ap.add_argument("--target_total",   type=int, default=9000)
    ap.add_argument("--general_budget", type=int, default=2000)
    ap.add_argument("--medical_budget", type=int, default=3000)
    ap.add_argument("--bench_budget",   type=int, default=1250)
    ap.add_argument("--threshold",      type=float, default=0.6,
                    help="mean_score < threshold ⇒ unmastered (stage-4 coding pool).")
    ap.add_argument("--seed",           type=int, default=5051)
    args = ap.parse_args()

    print(f"[build_medical_qa_stage5_retention]  target_total={args.target_total}  seed={args.seed}")
    print(f"  general={args.general_budget} medical={args.medical_budget} "
          f"bench={args.bench_budget} threshold={args.threshold}")

    # ----- Load caches -----
    with open(args.medical_cache) as f:
        medical_cache = json.load(f)
    with open(args.bench_cache) as f:
        bench_cache = json.load(f)
    with open(args.stage4_cache) as f:
        stage4_cache = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Phase 1: sample each pool -----
    print("\n=== Phase 1: sample pools ===")
    print("- General:")
    general_idxs = _sample_general(args.general_dir, args.general_budget, args.seed)
    print("- Medical:")
    medical_idxs = _sample_medical(medical_cache, args.medical_budget, args.seed + 1)
    print(f"  sampled: {len(medical_idxs)}")
    print("- Bench:")
    bench_idxs = _sample_bench(bench_cache, stage4_cache, args.bench_budget, args.seed + 2)
    print(f"  sampled: {len(bench_idxs)}")
    print("- Coding unmastered:")
    coding_idxs = _coding_unmastered_indices(stage4_cache, args.threshold)

    pool_sizes = {
        "general":           args.general_budget,
        "medical":           len(medical_idxs),
        "bench":             len(bench_idxs),
        "coding_unmastered": len(coding_idxs),
    }
    drawn_so_far = sum(pool_sizes.values())
    print(f"\n  drawn so far: {drawn_so_far} / target {args.target_total}")

    topup_idxs: list[int] = []
    if drawn_so_far < args.target_total:
        needed = args.target_total - drawn_so_far
        print(f"- Top-up medical UNSEEN (need {needed}):")
        topup_idxs = _topup_medical(medical_cache, set(medical_idxs), needed,
                                    args.seed + 3)
        pool_sizes["medical_topup"] = len(topup_idxs)
    else:
        print(f"  no top-up needed (drawn {drawn_so_far} ≥ {args.target_total})")

    grand_total = sum(pool_sizes.values())
    print(f"  GRAND TOTAL: {grand_total}  ({pool_sizes})")

    # ----- Phase 2: slice each pool -----
    print("\n=== Phase 2: slice each pool ===")
    pool_tables: list[tuple[str, pa.Table]] = []
    for fn, idxs in general_idxs.items():
        path = os.path.join(args.general_dir, fn)
        t = _drop_non_canonical_columns(_take(path, idxs))
        pool_tables.append((f"general/{fn}", t))
        print(f"  general/{fn}: {t.num_rows} rows  "
              f"data_source={dict(Counter(t.column('data_source').to_pylist()))}")

    med_t = _drop_non_canonical_columns(_take(args.medical_source, medical_idxs))
    pool_tables.append(("medical", med_t))
    print(f"  medical: {med_t.num_rows} rows  "
          f"data_source={dict(Counter(med_t.column('data_source').to_pylist()))}")

    bench_t = _drop_non_canonical_columns(_take(args.bench_source, bench_idxs))
    pool_tables.append(("bench", bench_t))
    print(f"  bench: {bench_t.num_rows} rows  "
          f"data_source={dict(Counter(bench_t.column('data_source').to_pylist()))}")

    coding_t = _drop_non_canonical_columns(_take(args.coding_source, coding_idxs))
    pool_tables.append(("coding_unmastered", coding_t))
    print(f"  coding_unmastered: {coding_t.num_rows} rows  "
          f"data_source={dict(Counter(coding_t.column('data_source').to_pylist()))}")

    if topup_idxs:
        top_t = _drop_non_canonical_columns(_take(args.medical_source, topup_idxs))
        pool_tables.append(("medical_topup", top_t))
        print(f"  medical_topup: {top_t.num_rows} rows  "
              f"data_source={dict(Counter(top_t.column('data_source').to_pylist()))}")

    # ----- Phase 3: schema reconciliation -----
    print("\n=== Phase 3: schema reconciliation ===")
    ei_types = [t.schema.field("extra_info").type for _, t in pool_tables]
    unified_ei = _build_super_extra_info_type(*ei_types)
    print(f"  unified extra_info: {unified_ei.num_fields} fields")
    print(f"  fields: {[f.name for f in unified_ei]}")

    reconciled = []
    for label, t in pool_tables:
        r = _reconcile(t, unified_ei)
        reconciled.append(r)
        print(f"  reconciled {label}: {r.num_rows} rows")

    # ----- Phase 4: concat + write -----
    print("\n=== Phase 4: concat + write ===")
    final = pa.concat_tables(reconciled, promote_options="default")
    print(f"  total rows: {final.num_rows}")
    print(f"  data_source counts: {dict(Counter(final.column('data_source').to_pylist()))}")
    tt_counts = Counter(
        (ei or {}).get("task_type") for ei in final.column("extra_info").to_pylist()
    )
    print(f"  task_type counts: {dict(tt_counts)}")

    # Sanity
    for r in final.column("reward_model").to_pylist()[:5]:
        assert isinstance(r["ground_truth"], str), \
            f"ground_truth must be str, got {type(r['ground_truth']).__name__}"

    out_path = os.path.join(args.output_dir, "retention.parquet")
    pq.write_table(final, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  wrote {out_path}  ({size_mb:.1f} MB)")

    # ----- Phase 5: write report -----
    report = {
        "params": {
            "target_total":   args.target_total,
            "general_budget": args.general_budget,
            "medical_budget": args.medical_budget,
            "bench_budget":   args.bench_budget,
            "threshold":      args.threshold,
            "seed":           args.seed,
            "seed_offsets":   {"general": 0, "medical": 1, "bench": 2, "topup": 3},
        },
        "pool_sizes":            pool_sizes,
        "final_rows":            final.num_rows,
        "data_source_counts":    dict(Counter(final.column("data_source").to_pylist())),
        "task_type_counts":      {str(k): v for k, v in tt_counts.items()},
        "unified_extra_info":    [f.name for f in unified_ei],
        "output_path":           out_path,
        "sources": {
            "general_dir":     args.general_dir,
            "medical_source":  args.medical_source,
            "bench_source":    args.bench_source,
            "coding_source":   args.coding_source,
        },
        "caches": {
            "medical_cache": args.medical_cache,
            "bench_cache":   args.bench_cache,
            "stage4_cache":  args.stage4_cache,
        },
    }
    report_path = os.path.join(args.output_dir, "_retention_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  wrote {report_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
