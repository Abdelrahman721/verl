"""Build the stage-9 retention pool.

Three pools per user spec (seed family = 9051):

  1. General  (6 000 rows total)
       - Chat   2 000 (seed=9051)
       - IFEval 2 000 (seed=9052)
       - Safety 2 000 (seed=9053)
       Flat per-source budgets — NOT the proportional / safety=50% rule
       used in stages 5/6/7.

  2. Medical_qa UNSEEN  (10 000 rows, seed=9054)
       UNSEEN vs (stage-1 ∪ stage-2 rollouts) against
       data/medical_qa/train.parquet (316 287 rows).
       Stage-1 seen indices come from
         data/medical_qa_stage3/_stats_cache.json["stage1"]["seen_source_indices"]
       Stage-2 seen indices come from the stage-9 stats cache
         data/medical_qa_stage9/_stage2_medqa_seen_cache.json
       (built by stats_medical_qa_stage9_retention.py, which re-walks
       medical_qa_stage2_rollouts and matches each prompt against the
       same 316 287-row medical_qa source so the indices are comparable
       with stage-1's set).

  3. Bench UNSEEN  (take all, ~6 583 rows)
       All indices in
         data/medical_qa_stage4/_bench_retention_cache.json["unseen_bench_source_indices"]
       sliced from data/medical_qa_stage3/train.parquet (the 17 193 bench
       rows are a strict subset of its 21 193 total rows; the cache
       indices are GLOBAL into that parquet).

Output:
  data/medical_qa_stage9/retention.parquet
  data/medical_qa_stage9/_retention_report.json

Reuses verbatim from earlier builds:
  - _drop_non_canonical_columns / _REWARD_MODEL_CANONICAL_TYPE  (stage-4 build)
  - _coerce_struct_column                                       (build_medical_qa_next)
  - _promote_strings_to_large                                   (build_curated_subset)
  - _build_super_extra_info_type / _normalise_reward_model      (stage-4 retention)

Usage:
    python -m full_mix.curriculum.build_medical_qa_stage9_retention \\
        --output_dir /data/abdelrahman/verl/data/medical_qa_stage9/
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


# Canonical prompt struct order — must match the stage-9 train build so both
# parquets concat cleanly later.
_PROMPT_CANONICAL_TYPE = pa.list_(pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
]))


# ============================================================================
# Pool 1 — General (per-source flat budgets)
# ============================================================================
def _sample_one_source(parquet_path: str, budget: int, seed: int) -> list[int]:
    """Random sample of `budget` row indices from `parquet_path`."""
    n = pq.read_metadata(parquet_path).num_rows
    if budget >= n:
        print(f"  WARNING: budget {budget} >= source {n}; taking all")
        return list(range(n))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), budget))


# ============================================================================
# Pool 2 — Medical_qa UNSEEN (vs stage-1 ∪ stage-2 rollouts)
# ============================================================================
def _sample_medical_unseen(stage3_cache: dict, stage9_cache: dict,
                            src_total: int, budget: int, seed: int) -> tuple[list[int], int, int, int]:
    """Return (sampled_indices, n_stage1_seen, n_stage2_seen, n_unseen)."""
    stage1_seen = set(stage3_cache["stage1"]["seen_source_indices"])
    stage2_seen = set(stage9_cache["seen_medqa_source_indices"])
    seen_union  = stage1_seen | stage2_seen
    unseen      = sorted(set(range(src_total)) - seen_union)
    print(f"  medical source rows: {src_total}")
    print(f"  stage-1 seen: {len(stage1_seen)}   stage-2 seen: {len(stage2_seen)}")
    print(f"  union seen:   {len(seen_union)}    unseen: {len(unseen)}")
    if budget >= len(unseen):
        print(f"  WARNING: budget {budget} >= unseen pool {len(unseen)}; taking all")
        return unseen, len(stage1_seen), len(stage2_seen), len(unseen)
    rng = random.Random(seed)
    return sorted(rng.sample(unseen, budget)), len(stage1_seen), len(stage2_seen), len(unseen)


# ============================================================================
# Pool 3 — Bench UNSEEN (take all)
# ============================================================================
def _bench_unseen_all(bench_cache: dict) -> list[int]:
    """All bench source indices not seen in medical_qa_benchmark_rollouts."""
    idxs = sorted(bench_cache["unseen_bench_source_indices"])
    print(f"  bench unseen (take all): {len(idxs)} rows")
    return idxs


# ============================================================================
# Helpers
# ============================================================================
def _take(parquet_path: str, indices: list[int]) -> pa.Table:
    t = pq.read_table(parquet_path)
    return t.take(pa.array(indices, type=pa.int64()))


def _normalise_prompt_struct(table: pa.Table) -> pa.Table:
    """Canonicalise prompt struct order to (role, content)."""
    if table.schema.field("prompt").type == _PROMPT_CANONICAL_TYPE:
        return table
    py = table.column("prompt").to_pylist()
    py = [[{"role": m.get("role", "") or "", "content": m.get("content", "") or ""}
           for m in conv] for conv in py]
    arr = pa.array(py, type=_PROMPT_CANONICAL_TYPE)
    idx = table.column_names.index("prompt")
    cols = list(table.columns)
    cols[idx] = arr
    return pa.table(cols, names=table.column_names)


def _reconcile(t: pa.Table, unified_ei: pa.StructType) -> pa.Table:
    t = _normalise_prompt_struct(t)
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
    ap.add_argument("--general_dir",
                    default="/data/abdelrahman/verl/data/full_mix/train")
    ap.add_argument("--medical_source",
                    default="/data/abdelrahman/verl/data/medical_qa/train.parquet")
    ap.add_argument("--bench_source",
                    default="/data/abdelrahman/verl/data/medical_qa_stage3/train.parquet")
    ap.add_argument("--stage3_cache",
                    default="/data/abdelrahman/verl/data/medical_qa_stage3/_stats_cache.json")
    ap.add_argument("--stage9_cache",
                    default="/data/abdelrahman/verl/data/medical_qa_stage9/_stage2_medqa_seen_cache.json")
    ap.add_argument("--bench_cache",
                    default="/data/abdelrahman/verl/data/medical_qa_stage4/_bench_retention_cache.json")
    ap.add_argument("--chat_budget",   type=int, default=2000)
    ap.add_argument("--ifeval_budget", type=int, default=2000)
    ap.add_argument("--safety_budget", type=int, default=2000)
    ap.add_argument("--medical_budget", type=int, default=10000)
    ap.add_argument("--seed",          type=int, default=9051)
    args = ap.parse_args()

    print(f"[build_medical_qa_stage9_retention]  seed={args.seed}")
    print(f"  chat={args.chat_budget} ifeval={args.ifeval_budget} "
          f"safety={args.safety_budget}  medical={args.medical_budget}  "
          f"bench=take-all")

    # ----- Load caches -----
    with open(args.stage3_cache) as f:
        stage3_cache = json.load(f)
    with open(args.stage9_cache) as f:
        stage9_cache = json.load(f)
    with open(args.bench_cache) as f:
        bench_cache = json.load(f)
    src_total = stage3_cache["stage1"]["src_total_rows"]

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Phase 1: sample pools -----
    print("\n=== Phase 1: sample pools ===")
    print("- General Chat:")
    chat_idxs   = _sample_one_source(
        os.path.join(args.general_dir, "chat_with_baseline_train.parquet"),
        args.chat_budget,  args.seed)
    print("- General IFEval:")
    ifeval_idxs = _sample_one_source(
        os.path.join(args.general_dir, "ifeval_train.parquet"),
        args.ifeval_budget, args.seed + 1)
    print("- General Safety:")
    safety_idxs = _sample_one_source(
        os.path.join(args.general_dir, "safety_train.parquet"),
        args.safety_budget, args.seed + 2)
    print("- Medical_qa UNSEEN:")
    medical_idxs, n_s1, n_s2, n_unseen = _sample_medical_unseen(
        stage3_cache, stage9_cache, src_total, args.medical_budget, args.seed + 3)
    print("- Bench UNSEEN (take all):")
    bench_idxs = _bench_unseen_all(bench_cache)

    pool_sizes = {
        "general_chat":     len(chat_idxs),
        "general_ifeval":   len(ifeval_idxs),
        "general_safety":   len(safety_idxs),
        "medical_unseen":   len(medical_idxs),
        "bench_unseen":     len(bench_idxs),
    }
    grand_total = sum(pool_sizes.values())
    print(f"\n  GRAND TOTAL: {grand_total}  ({pool_sizes})")

    # ----- Phase 2: slice each pool -----
    print("\n=== Phase 2: slice each pool ===")
    pool_tables: list[tuple[str, pa.Table]] = []

    chat_t = _drop_non_canonical_columns(_take(
        os.path.join(args.general_dir, "chat_with_baseline_train.parquet"), chat_idxs))
    pool_tables.append(("general/chat", chat_t))
    print(f"  general/chat: {chat_t.num_rows} rows  "
          f"data_source={dict(Counter(chat_t.column('data_source').to_pylist()))}")

    if_t = _drop_non_canonical_columns(_take(
        os.path.join(args.general_dir, "ifeval_train.parquet"), ifeval_idxs))
    pool_tables.append(("general/ifeval", if_t))
    print(f"  general/ifeval: {if_t.num_rows} rows  "
          f"data_source={dict(Counter(if_t.column('data_source').to_pylist()))}")

    sf_t = _drop_non_canonical_columns(_take(
        os.path.join(args.general_dir, "safety_train.parquet"), safety_idxs))
    pool_tables.append(("general/safety", sf_t))
    print(f"  general/safety: {sf_t.num_rows} rows  "
          f"data_source={dict(Counter(sf_t.column('data_source').to_pylist()))}")

    med_t = _drop_non_canonical_columns(_take(args.medical_source, medical_idxs))
    pool_tables.append(("medical", med_t))
    print(f"  medical: {med_t.num_rows} rows  "
          f"data_source={dict(Counter(med_t.column('data_source').to_pylist()))}")

    bench_t = _drop_non_canonical_columns(_take(args.bench_source, bench_idxs))
    pool_tables.append(("bench", bench_t))
    print(f"  bench: {bench_t.num_rows} rows  "
          f"data_source={dict(Counter(bench_t.column('data_source').to_pylist()))}")

    # ----- Phase 3: schema reconciliation -----
    print("\n=== Phase 3: schema reconciliation ===")
    ei_types = [t.schema.field("extra_info").type for _, t in pool_tables]
    unified_ei = _build_super_extra_info_type(*ei_types)
    print(f"  unified extra_info: {unified_ei.num_fields} fields")

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
            "chat_budget":     args.chat_budget,
            "ifeval_budget":   args.ifeval_budget,
            "safety_budget":   args.safety_budget,
            "medical_budget":  args.medical_budget,
            "seed":            args.seed,
            "seed_offsets":    {"chat": 0, "ifeval": 1, "safety": 2, "medical": 3},
        },
        "medical_seen_breakdown": {
            "stage1_seen":  n_s1,
            "stage2_seen":  n_s2,
            "union_seen":   n_s1 + n_s2 - max(0, n_s1 + n_s2 - (src_total - n_unseen)),
            "unseen_pool":  n_unseen,
        },
        "pool_sizes":         pool_sizes,
        "final_rows":         final.num_rows,
        "data_source_counts": dict(Counter(final.column("data_source").to_pylist())),
        "task_type_counts":   {str(k): v for k, v in tt_counts.items()},
        "unified_extra_info": [f.name for f in unified_ei],
        "output_path":        out_path,
    }
    report_path = os.path.join(args.output_dir, "_retention_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  wrote {report_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
