"""Build the stage-6 retention pool.

Three pools per user spec (seed family = 6051):

  1. General        (405 rows, seed=6051)
        Safety = 50 % of general = 203 rows (largest-remainder rounded).
        Chat + IF = 202 rows split proportional to source size.
        Sources: data/full_mix/train/{chat_with_baseline,ifeval,safety}_train.parquet

  2. Medical_qa UNSEEN (608 rows, seed=6052)
        UNSEEN vs medical_qa_rollouts (stage-3 cache: 283 749 unseen indices).
        Source: data/medical_qa/train.parquet

  3. Stage-5 coding UNMASTERED (cap 1 000, seed=6053)
        mean_score < threshold from stage-5 coding rollouts.
        If pool > 1 000, rng.sample(pool, 1 000).
        Source: data/medical_qa_stage5/train.parquet (35 120 coding rows).

Output:
  data/medical_qa_stage6/retention.parquet
  data/medical_qa_stage6/_retention_report.json

Reuses verbatim from earlier builds:
  - _drop_non_canonical_columns / _REWARD_MODEL_CANONICAL_TYPE  (stage-4 build)
  - _coerce_struct_column                                       (build_medical_qa_next)
  - _promote_strings_to_large                                   (build_curated_subset)
  - _build_super_extra_info_type / _normalise_reward_model      (stage-4 retention)
  - _sample_medical / _topup_medical seed pattern               (stage-5 retention)

Usage:
    python -m full_mix.curriculum.build_medical_qa_stage6_retention \\
        --output_dir /data/abdelrahman/verl/data/medical_qa_stage6/
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

# Canonical prompt struct order — must match the stage-6 train build so both
# parquets concat cleanly later.
_PROMPT_CANONICAL_TYPE = pa.list_(pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
]))


# ============================================================================
# Pool 1 — General (Safety = 50%; Chat + IF proportional)
# ============================================================================
def _sample_general(general_dir: str, budget: int, seed: int) -> dict[str, list[int]]:
    """Safety = 50% of budget. Chat + IF = remaining 50% split proportionally."""
    sizes = {fn: pq.read_metadata(os.path.join(general_dir, fn)).num_rows
             for fn in _GENERAL_FILES}

    safety_n = round(budget * 0.5)
    remainder = budget - safety_n
    # Proportional split of remainder between chat + ifeval
    chat_size = sizes["chat_with_baseline_train.parquet"]
    if_size   = sizes["ifeval_train.parquet"]
    total_ci  = chat_size + if_size
    chat_n_f  = remainder * chat_size / total_ci
    if_n_f    = remainder * if_size   / total_ci
    chat_n    = int(chat_n_f)
    if_n      = int(if_n_f)
    # Largest-remainder rounding so chat_n + if_n == remainder exactly
    leftover = remainder - chat_n - if_n
    if leftover > 0:
        # Give to the one with the largest fractional remainder
        if (chat_n_f - chat_n) >= (if_n_f - if_n):
            chat_n += leftover
        else:
            if_n += leftover

    counts = {
        "chat_with_baseline_train.parquet": chat_n,
        "ifeval_train.parquet":             if_n,
        "safety_train.parquet":             safety_n,
    }
    print(f"  general sizes={sizes}  → counts={counts}  (sum={sum(counts.values())})")
    assert sum(counts.values()) == budget, "general split must equal budget"

    rng = random.Random(seed)
    return {
        fn: sorted(rng.sample(range(sizes[fn]), n) if n < sizes[fn] else list(range(sizes[fn])))
        for fn, n in counts.items()
    }


# ============================================================================
# Pool 2 — Medical UNSEEN (vs medical_qa_rollouts)
# ============================================================================
def _medical_unseen(medical_cache: dict) -> list[int]:
    stage1 = medical_cache["stage1"]
    seen = set(stage1["seen_source_indices"])
    total = stage1["src_total_rows"]
    unseen = sorted(set(range(total)) - seen)
    print(f"  medical source rows: {total}  seen: {len(seen)}  unseen: {len(unseen)}")
    return unseen


def _sample_medical(medical_cache: dict, budget: int, seed: int) -> list[int]:
    unseen = _medical_unseen(medical_cache)
    if budget >= len(unseen):
        print(f"  WARNING: budget {budget} >= unseen pool {len(unseen)}; taking all")
        return unseen
    rng = random.Random(seed)
    return sorted(rng.sample(unseen, budget))


# ============================================================================
# Pool 3 — Stage-5 coding UNMASTERED (cap)
# ============================================================================
def _coding_unmastered(stage5_cache: dict, threshold: float,
                       cap: int, seed: int) -> tuple[list[int], int]:
    """Returns (kept_indices, raw_count_before_cap)."""
    _CODING_EMS = frozenset({
        "icd10_multilabel", "icd10_instruction_follow",
        "snomed_multilabel", "snomed_instruction_follow",
    })
    full_pool = sorted({
        p["coding_source_index"] for p in stage5_cache["per_prompt"]
        if p["coding_source_index"] is not None
        and p["mean_score"] < threshold
        and p["eval_mode"] and all(em in _CODING_EMS for em in p["eval_mode"])
    })
    raw_count = len(full_pool)
    print(f"  stage-5 coding-unmastered (mean_score < {threshold}): {raw_count} rows")
    if raw_count <= cap:
        print(f"  pool ≤ cap ({cap}); taking all {raw_count}")
        return full_pool, raw_count
    rng = random.Random(seed)
    kept = sorted(rng.sample(full_pool, cap))
    print(f"  pool > cap ({cap}); subsampled to {len(kept)} (seed={seed})")
    return kept, raw_count


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
    ap.add_argument("--stage5_coding_source",
                    default="/data/abdelrahman/verl/data/medical_qa_stage5/train.parquet")
    ap.add_argument("--medical_cache",
                    default="/data/abdelrahman/verl/data/medical_qa_stage3/_stats_cache.json")
    ap.add_argument("--stage5_cache",
                    default="/data/abdelrahman/verl/data/medical_qa_stage6/_stage5_retention_cache.json")
    ap.add_argument("--general_budget", type=int, default=405)
    ap.add_argument("--medical_budget", type=int, default=608)
    ap.add_argument("--unmastered_cap", type=int, default=1000)
    ap.add_argument("--threshold",      type=float, default=0.6)
    ap.add_argument("--seed",           type=int, default=6051)
    args = ap.parse_args()

    print(f"[build_medical_qa_stage6_retention]  seed={args.seed}")
    print(f"  general={args.general_budget}  medical={args.medical_budget}  "
          f"unmastered_cap={args.unmastered_cap}  threshold={args.threshold}")

    # ----- Load caches -----
    with open(args.medical_cache) as f:
        medical_cache = json.load(f)
    with open(args.stage5_cache) as f:
        stage5_cache = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Phase 1: sample pools -----
    print("\n=== Phase 1: sample pools ===")
    print("- General:")
    general_idxs = _sample_general(args.general_dir, args.general_budget, args.seed)
    print("- Medical:")
    medical_idxs = _sample_medical(medical_cache, args.medical_budget, args.seed + 1)
    print(f"  sampled: {len(medical_idxs)}")
    print("- Stage-5 unmastered:")
    unmastered_idxs, unmastered_raw = _coding_unmastered(
        stage5_cache, args.threshold, args.unmastered_cap, args.seed + 2,
    )

    pool_sizes = {
        "general":             args.general_budget,
        "medical":             len(medical_idxs),
        "unmastered_kept":     len(unmastered_idxs),
        "unmastered_raw":      unmastered_raw,
    }
    grand_total = pool_sizes["general"] + pool_sizes["medical"] + pool_sizes["unmastered_kept"]
    print(f"\n  GRAND TOTAL: {grand_total}  ({pool_sizes})")

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

    if unmastered_idxs:
        un_t = _drop_non_canonical_columns(_take(args.stage5_coding_source, unmastered_idxs))
        pool_tables.append(("stage5_unmastered", un_t))
        print(f"  stage5_unmastered: {un_t.num_rows} rows  "
              f"data_source={dict(Counter(un_t.column('data_source').to_pylist()))}")

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
            "general_budget":  args.general_budget,
            "medical_budget":  args.medical_budget,
            "unmastered_cap":  args.unmastered_cap,
            "threshold":       args.threshold,
            "seed":            args.seed,
            "seed_offsets":    {"general": 0, "medical": 1, "unmastered": 2},
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
