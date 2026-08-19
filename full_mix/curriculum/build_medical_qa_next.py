"""Assemble the next medical_qa training-stage parquet.

Consumes the cache emitted by `stats_medical_qa_next.py` and writes:
  {output_dir}/train.parquet           — the actual training file
  {output_dir}/_build_report.json      — per-pool counts + parameters
  {output_dir}/seen_prompts_manifest.json — source-row indices the model has
                                            already been rolled out on (carry
                                            from the input cache + the rows
                                            this build adds). Future stages
                                            read this to dedupe before
                                            sampling.

Four pools are concatenated into the output parquet:
  A. CARRY-FORWARD  — medical_qa/train.parquet rows whose mean rollout score
                      was below --threshold.
  B. UNSEEN CONV    — medical_conv rows in medical_qa/train.parquet that did
                      not appear in any rollout. Sampled to --conv_keep_n.
  C. CARVE CLINICAL — carve_clinical_reasoning.parquet rows that survive
                      seen-rollout dedup. Schema-converted to match
                      medical_qa/train.parquet (struct ground_truth →
                      JSON string via medical_qa_compat.transform_table).
  D. CARVE LAY      — carve_lay_patient_tier.parquet, same treatment as C.

Usage:
    python -m full_mix.curriculum.build_medical_qa_next \\
        --cache /data/abdelrahman/verl/data/medical_qa_stage2/_stats_cache.json \\
        --threshold 0.6 \\
        --conv_keep_n 5000 \\
        --output /data/abdelrahman/verl/data/medical_qa_stage2/

Reused from elsewhere in the repo:
  - full_mix.preprocess.medical_qa_compat.transform_table  (struct → JSON string)
  - full_mix.curriculum.build_curated_subset._promote_strings_to_large
    (large_string promotion so verl's HF Datasets loader can concat shards)
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

# Reuse the existing helpers — no copy-paste.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FULL_MIX_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_FULL_MIX_DIR)
for _p in (_REPO_ROOT, _FULL_MIX_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from full_mix.preprocess.medical_qa_compat import transform_table  # noqa: E402
from full_mix.curriculum.build_curated_subset import _promote_strings_to_large  # noqa: E402


# ============================================================================
# Pool assembly
# ============================================================================
def _take(parquet_path: str, indices: list[int]) -> pa.Table:
    """Read the source parquet once and slice it by row index."""
    t = pq.read_table(parquet_path)
    return t.take(pa.array(indices, type=pa.int64()))


def _carry_forward_indices(per_prompt, threshold: float) -> tuple[list[int], list[int]]:
    """Split per_prompt entries by mean_score vs threshold. Returns (carry, mastered)."""
    carry, mastered = [], []
    for p in per_prompt:
        if p["source_index"] is None:
            continue
        (carry if p["mean_score"] < threshold else mastered).append(p["source_index"])
    return sorted(carry), sorted(mastered)


def _sample_conv(unseen_indices: list[int], keep_n: int, seed: int) -> list[int]:
    """Sample `keep_n` indices from `unseen_indices` reproducibly."""
    if keep_n >= len(unseen_indices):
        if keep_n > len(unseen_indices):
            print(f"  WARNING: --conv_keep_n={keep_n} > pool size {len(unseen_indices)};"
                  f" taking all available.")
        return sorted(unseen_indices)
    rng = random.Random(seed)
    return sorted(rng.sample(unseen_indices, keep_n))


def _coerce_struct_column(table: pa.Table, col_name: str, target_type: pa.DataType) -> pa.Table:
    """Replace a struct column with one re-built under `target_type`.

    pyarrow can't directly cast struct columns when the source has `null`
    typed fields where the target expects `string`/`int64` (carve files vs
    medical_qa source). Round-tripping through pylist coerces each field
    correctly: missing fields → None, int-valued doubles → int, etc.
    """
    if col_name not in table.column_names:
        return table
    records = table.column(col_name).to_pylist()
    # Cast int-shaped floats in fields that should be ints. pa.array does this
    # automatically when the target field type is int and the value is an int
    # already; for floats like 1.0 we explicitly truncate to avoid pa raising.
    int_fields = {f.name for f in target_type if pa.types.is_integer(f.type)}
    if int_fields:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            for k in int_fields:
                v = rec.get(k)
                if isinstance(v, float):
                    rec[k] = int(v) if v == int(v) else v  # let pa raise on truly fractional
    new_arr = pa.array(records, type=target_type)
    idx = table.column_names.index(col_name)
    cols = list(table.columns)
    cols[idx] = new_arr
    schema = pa.schema(
        [pa.field(n, t, table.schema.field(n).nullable, table.schema.field(n).metadata)
         if n != col_name else pa.field(col_name, target_type)
         for n, t in zip(table.column_names, [c.type for c in cols])]
    )
    return pa.table(cols, schema=schema)


def _reconcile_schema(tables: list[pa.Table]) -> list[pa.Table]:
    """Bring every pool to a single schema before concat.

    Steps:
      1. Pick the FIRST table's schema as canonical (this is pool A — sliced
         from the medical_qa source — so it carries the medical_qa_compat
         shape we want for the output parquet).
      2. Coerce the `extra_info` and `reward_model` struct columns of every
         other pool to the canonical struct types via a pylist round-trip.
         Carve files come from raw hazem source and have a subtly different
         extra_info struct (`null`-typed fields where source has `string`,
         `double` where source has `int64`).
      3. Promote every top-level string column to large_string for verl's
         HF Datasets loader (same rule as build_curated_subset).
    """
    target_schema = tables[0].schema
    out = []
    for i, t in enumerate(tables):
        if i > 0:
            for name in ("extra_info", "reward_model"):
                if name in t.column_names:
                    field = target_schema.field(name)
                    if t.schema.field(name).type != field.type:
                        t = _coerce_struct_column(t, name, field.type)
        promoted = _promote_strings_to_large(t.schema)
        if promoted != t.schema:
            t = t.cast(promoted)
        out.append(t)
    return out


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache", required=True,
                    help="Path to _stats_cache.json from stats_medical_qa_next.py.")
    ap.add_argument("--threshold", type=float, required=True,
                    help="Mean rollout `score` below which a prompt is carried forward.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--conv_keep_n", type=int,
                   help="Absolute number of UNSEEN conv prompts to sample.")
    g.add_argument("--conv_keep_fraction", type=float,
                   help="Fraction (0,1] of the UNSEEN conv pool to sample.")
    ap.add_argument("--output", required=True,
                    help="Output directory (train.parquet + reports land here).")
    ap.add_argument("--seed", type=int, default=1234,
                    help="Sampling seed for the conv pool (default 1234).")
    args = ap.parse_args()

    print(f"[build_medical_qa_next] cache     = {args.cache}")
    print(f"[build_medical_qa_next] threshold = {args.threshold}")
    print(f"[build_medical_qa_next] output    = {args.output}")
    print(f"[build_medical_qa_next] seed      = {args.seed}")

    # ----- Load cache -----
    with open(args.cache) as f:
        cache = json.load(f)
    source_parquet = cache["meta"]["source_parquet"]
    print(f"\nsource parquet (from cache): {source_parquet}")
    print(f"n distinct medical prompts seen: {cache['meta']['n_distinct_medical_prompts']}")
    print(f"n orphans: {cache['meta']['n_orphans']}")

    # ----- Pool A: carry forward -----
    print("\n=== Pool A — carry forward (medical_qa + medical_conv hard) ===")
    carry_idx, mastered_idx = _carry_forward_indices(
        cache["per_prompt"], args.threshold,
    )
    print(f"  carry = {len(carry_idx)}   mastered = {len(mastered_idx)}")
    pool_a = _take(source_parquet, carry_idx)
    em_a = Counter(pool_a.column("data_source").to_pylist())
    print(f"  pool A rows: {pool_a.num_rows}   data_source: {dict(em_a)}")

    # ----- Pool B: unseen conv -----
    print("\n=== Pool B — unseen conv sample ===")
    unseen_conv = cache["unseen_conv_source_indices"]
    keep_n = (args.conv_keep_n if args.conv_keep_n is not None
              else max(1, int(args.conv_keep_fraction * len(unseen_conv))))
    sampled_conv = _sample_conv(unseen_conv, keep_n, args.seed)
    print(f"  pool size: {len(unseen_conv)}   target: {keep_n}   sampled: {len(sampled_conv)}")
    pool_b = _take(source_parquet, sampled_conv)
    em_b = Counter(pool_b.column("data_source").to_pylist())
    print(f"  pool B rows: {pool_b.num_rows}   data_source: {dict(em_b)}")
    assert all(ds == "medical_conv" for ds in pool_b.column("data_source").to_pylist()), \
        "pool B contains non-medical_conv rows"

    # ----- Pools C/D: carve files (apply transform_table for schema compat) -----
    print("\n=== Pools C/D — carve files (raw → medical_qa_compat schema) ===")
    carve_tables = []
    for cp, summary in cache["carve"].items():
        idxs = summary["after_dedup_carve_row_indices"]
        raw = _take(cp, idxs)
        # Raw hazem parquet has reward_model.ground_truth as struct — convert.
        compat = transform_table(raw)
        em = Counter(compat.column("data_source").to_pylist())
        print(f"  {os.path.basename(cp)}: rows={compat.num_rows}   data_source: {dict(em)}")
        carve_tables.append((cp, compat))

    # ----- Reconcile schemas + concat -----
    print("\n=== Schema reconciliation + concat ===")
    all_tables = [pool_a, pool_b] + [t for _, t in carve_tables]
    print("  per-pool rows: " + " + ".join(str(t.num_rows) for t in all_tables))
    reconciled = _reconcile_schema(all_tables)
    try:
        final = pa.concat_tables(reconciled, promote_options="default")
    except pa.lib.ArrowInvalid as e:
        # If schemas still don't line up, print a per-pool field diff before failing.
        print("  concat failed; per-pool schemas:")
        for i, t in enumerate(reconciled):
            print(f"    pool {i}: {t.schema!r}")
        raise
    print(f"  total rows: {final.num_rows}")
    print(f"  final data_source counts: {dict(Counter(final.column('data_source').to_pylist()))}")

    # Sanity: every row's reward_model.ground_truth must be a string.
    rm_sample = final.column("reward_model").to_pylist()[:5]
    for r in rm_sample:
        assert isinstance(r.get("ground_truth"), str), \
            f"ground_truth not a string: {type(r.get('ground_truth')).__name__}"
    print("  ground_truth is str for first 5 rows ✓")

    # ----- Write outputs -----
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.parquet")
    pq.write_table(final, train_path)
    print(f"\nwrote {train_path}  ({os.path.getsize(train_path) / (1024*1024):.1f} MB)")

    # Build report
    report = {
        "cache": args.cache,
        "threshold": args.threshold,
        "conv_keep_n_requested": args.conv_keep_n,
        "conv_keep_fraction_requested": args.conv_keep_fraction,
        "conv_keep_seed": args.seed,
        "pools": {
            "A_carry_forward": {
                "rows": pool_a.num_rows,
                "data_source": dict(em_a),
                "source_indices_count": len(carry_idx),
            },
            "B_unseen_conv_sample": {
                "rows": pool_b.num_rows,
                "source_indices_count": len(sampled_conv),
                "unseen_pool_total": len(unseen_conv),
            },
        },
        "carve": {
            os.path.basename(cp): {
                "rows": t.num_rows,
                "after_dedup_total": len(cache["carve"][cp]["after_dedup_carve_row_indices"]),
            }
            for cp, t in carve_tables
        },
        "output_train_parquet": train_path,
        "total_rows": final.num_rows,
        "total_data_source_counts": dict(Counter(final.column("data_source").to_pylist())),
        "mastered_source_indices_count": len(mastered_idx),
    }
    report_path = os.path.join(args.output, "_build_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")

    # Seen-prompts manifest (carry + mastered = everything sampled in the
    # previous stage; sampled_conv was NOT sampled before — they're being
    # introduced now and the model will see them in the next stage). The
    # manifest records source_indices in the medical_qa source parquet that
    # the model has been exposed to up to and including the previous stage.
    manifest = {
        "source_parquet": source_parquet,
        "stage_label": "after_previous_medical_qa_run",
        "seen_source_indices": sorted(set(carry_idx) | set(mastered_idx)),
        "comment": (
            "These row indices in source_parquet are the medical prompts the "
            "model was rolled out on during the previous training stage. "
            "Subsequent build steps should dedupe candidate pools against this set."
        ),
    }
    manifest_path = os.path.join(args.output, "seen_prompts_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}  ({len(manifest['seen_source_indices'])} indices)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
