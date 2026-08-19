"""Assemble the medical_qa STAGE-4 training parquet (medical-coding focus).

Two pools, no replay (the user will splice in retention separately):
  S — SNOMED single-label train (ground_truth already JSON string)
  I — ICD-10 multi-label train  (ground_truth is list<string> → JSON-serialise)

Each pool is stamped with `extra_info.task_type` so the
`qa_openrouter_bench._bench_determine_eval_mode` dispatcher routes to the
right scorer at training time. Schemas are reconciled to a unified shape and
all top-level string columns are promoted to `large_string` to match the
rest of the repo's parquet pipeline.

Usage:
    python -m full_mix.curriculum.build_medical_qa_stage4 \\
        --snomed /data/muhsen/repos/finalize-dataset/abdelrahman_rl/final_sl_subsampled.parquet \\
        --icd    /home/mbinakba/mcs-instruct-dataset/avey_icd_single_label_rl_train_dataset.parquet \\
        --output /data/abdelrahman/verl/data/medical_qa_stage4/
"""

from __future__ import annotations

import argparse
import json
import os
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


# ============================================================================
# Reconciliation targets
# ============================================================================
# All coding rows end up with this reward_model shape — matches the rest of
# the medical_qa_compat pipeline so the same training loader handles them.
_REWARD_MODEL_CANONICAL_TYPE = pa.struct([
    pa.field("style", pa.string()),
    pa.field("ground_truth", pa.string()),
])


def _build_unified_extra_info_type(snomed_ei: pa.StructType,
                                   icd_ei: pa.StructType) -> pa.StructType:
    """Union of all fields across SNOMED and ICD extra_info, plus `task_type`.

    Each pool's pylist round-trip (via `_coerce_struct_column`) fills missing
    keys with None automatically, so SNOMED rows get None for ICD-only fields
    (and vice versa).
    """
    seen: dict[str, pa.Field] = {}
    for src in (snomed_ei, icd_ei):
        for i in range(src.num_fields):
            f = src.field(i)
            if f.name not in seen:
                seen[f.name] = f
    # Strings as the canonical type for list-of-string ICD fields — the
    # pylist round-trip flattens single-element list<string> to its content,
    # but to be safe we cast such columns inline (see `_normalize_icd_extra_info`).
    fields = list(seen.values())
    if "task_type" not in seen:
        fields.append(pa.field("task_type", pa.string()))
    return pa.struct(fields)


def _stamp_task_type(table: pa.Table, task_type: str) -> pa.Table:
    """Set `extra_info.task_type = task_type` on every row of `table`."""
    records = table.column("extra_info").to_pylist()
    for r in records:
        if isinstance(r, dict):
            r["task_type"] = task_type
        # else: pa.array will coerce non-dicts as None — won't happen here
    # The target type still has `task_type` because the unified type includes
    # it; for now we keep the source type and let `_coerce_struct_column`
    # re-build the array later. To avoid a double round-trip, we replace the
    # column with one constructed at the *current* type augmented with
    # task_type:
    cur_type = table.schema.field("extra_info").type
    field_names = {f.name for f in cur_type}
    if "task_type" not in field_names:
        augmented = pa.struct(list(cur_type) + [pa.field("task_type", pa.string())])
    else:
        augmented = cur_type
    new_arr = pa.array(records, type=augmented)
    idx = table.column_names.index("extra_info")
    cols = list(table.columns)
    cols[idx] = new_arr
    return pa.table(cols, names=table.column_names)


def _icd_serialise_ground_truth(table: pa.Table) -> pa.Table:
    """Convert ICD `reward_model.ground_truth: list<string>` → JSON string.

    The ICD source carries the gold codes as a `list<string>`. `qa_openrouter_bench`'s
    ICD scorer accepts either a list or a JSON string (it `str(ground_truth)`s
    and runs a regex), but downstream tooling and DataProto concat expect a
    string. Serialise to JSON here so the rest of the pipeline is uniform.
    """
    records = table.column("reward_model").to_pylist()
    new_rm = []
    for r in records:
        codes = r.get("ground_truth") if isinstance(r, dict) else None
        # `codes` is a list[str] in the source; defensive cast handles None/str.
        if codes is None:
            gt_str = "[]"
        elif isinstance(codes, list):
            gt_str = json.dumps(list(codes), ensure_ascii=False)
        elif isinstance(codes, str):
            gt_str = codes   # already serialised
        else:
            gt_str = json.dumps(codes, ensure_ascii=False, default=str)
        new_rm.append({"style": r.get("style", "") if isinstance(r, dict) else "",
                       "ground_truth": gt_str})
    new_arr = pa.array(new_rm, type=_REWARD_MODEL_CANONICAL_TYPE)
    idx = table.column_names.index("reward_model")
    cols = list(table.columns)
    cols[idx] = new_arr
    return pa.table(cols, names=table.column_names)


def _drop_non_canonical_columns(t: pa.Table) -> pa.Table:
    """Keep only the canonical column set so concat doesn't choke on extras.

    Canonical = `data_source, prompt, ability, reward_model, extra_info`.
    The ICD source ships extra top-level columns (`code`, `description`) — we
    don't need them in stage-4 since `code` is already inside extra_info and
    description is unused by the scorer.
    """
    canonical = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    return t.select([n for n in canonical if n in t.column_names])


def _reconcile(t: pa.Table, unified_ei_type: pa.StructType) -> pa.Table:
    """Coerce extra_info + reward_model to canonical types and promote strings."""
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
    ap.add_argument("--snomed", required=True,
                    help="SNOMED single-label train parquet.")
    ap.add_argument("--icd", required=True,
                    help="ICD multi-label train parquet.")
    ap.add_argument("--output", required=True,
                    help="Output directory; writes train.parquet + _build_report.json.")
    args = ap.parse_args()

    print(f"[build_medical_qa_stage4]")
    print(f"  SNOMED : {args.snomed}")
    print(f"  ICD    : {args.icd}")
    print(f"  output : {args.output}")

    # ----- Phase 1: load + stamp task_type -----
    print("\n=== Phase 1: load + stamp task_type ===")
    snomed = pq.read_table(args.snomed)
    icd    = pq.read_table(args.icd)
    print(f"  SNOMED rows: {snomed.num_rows}")
    print(f"  ICD    rows: {icd.num_rows}")
    # Drop non-canonical top-level columns (ICD ships `code`, `description`)
    snomed = _drop_non_canonical_columns(snomed)
    icd    = _drop_non_canonical_columns(icd)
    # Stamp task_type on each pool
    snomed = _stamp_task_type(snomed, "snomed_single_label")
    icd    = _stamp_task_type(icd,    "icd_multi_label")
    print("  task_type stamped ✓")

    # ----- Phase 2: ICD ground_truth list → JSON string -----
    print("\n=== Phase 2: ICD ground_truth list → JSON string ===")
    icd = _icd_serialise_ground_truth(icd)
    # Quick sanity: pull first row and confirm
    sample = icd.column("reward_model").to_pylist()[0]
    print(f"  sample ICD ground_truth (type={type(sample['ground_truth']).__name__}): "
          f"{sample['ground_truth']!r}")

    # SNOMED's ground_truth is already a JSON string per the source schema.
    # Coerce the struct shape to the canonical {style: string, ground_truth: string}
    # using the same pylist round-trip (handles missing `style` field gracefully).
    snomed_rm_records = snomed.column("reward_model").to_pylist()
    snomed_rm = pa.array(
        [{"style": (r.get("style") if isinstance(r, dict) else "") or "",
          "ground_truth": (r.get("ground_truth") if isinstance(r, dict) else "")}
         for r in snomed_rm_records],
        type=_REWARD_MODEL_CANONICAL_TYPE,
    )
    idx = snomed.column_names.index("reward_model")
    cols = list(snomed.columns)
    cols[idx] = snomed_rm
    snomed = pa.table(cols, names=snomed.column_names)
    print("  SNOMED ground_truth normalised ✓")

    # ----- Phase 3: build the unified extra_info type -----
    print("\n=== Phase 3: schema reconciliation ===")
    snomed_ei_type = snomed.schema.field("extra_info").type
    icd_ei_type    = icd.schema.field("extra_info").type
    unified_ei_type = _build_unified_extra_info_type(snomed_ei_type, icd_ei_type)
    print(f"  SNOMED extra_info: {snomed_ei_type.num_fields} fields")
    print(f"  ICD    extra_info: {icd_ei_type.num_fields} fields")
    print(f"  unified extra_info: {unified_ei_type.num_fields} fields")

    # ----- Phase 4: reconcile each pool's schema -----
    snomed_r = _reconcile(snomed, unified_ei_type)
    icd_r    = _reconcile(icd,    unified_ei_type)
    print(f"  pool S (SNOMED) rows: {snomed_r.num_rows}   schema reconciled")
    print(f"  pool I (ICD)    rows: {icd_r.num_rows}      schema reconciled")

    # ----- Phase 5: concat -----
    print("\n=== Phase 5: concat ===")
    try:
        final = pa.concat_tables([snomed_r, icd_r], promote_options="default")
    except pa.lib.ArrowInvalid as e:
        print("concat failed; per-pool schemas:")
        for label, t in (("S (SNOMED)", snomed_r), ("I (ICD)", icd_r)):
            print(f"\n  {label}: {t.schema}")
        raise
    print(f"  total rows: {final.num_rows}")
    print(f"  data_source counts: {dict(Counter(final.column('data_source').to_pylist()))}")

    # Sanity probes
    rm_sample = final.column("reward_model").to_pylist()[:5]
    for r in rm_sample:
        assert isinstance(r.get("ground_truth"), str), \
            f"ground_truth must be str, got {type(r.get('ground_truth')).__name__}"
    print("  ground_truth is str for first 5 rows ✓")
    ei_sample = final.column("extra_info").to_pylist()[:5]
    for r in ei_sample:
        assert isinstance(r, dict) and r.get("task_type") in ("snomed_single_label", "icd_multi_label"), \
            f"task_type must be set on every row, got {r.get('task_type') if isinstance(r, dict) else type(r)}"
    print("  task_type populated on first 5 rows ✓")

    # ----- Phase 6: write outputs -----
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.parquet")
    pq.write_table(final, train_path)
    size_mb = os.path.getsize(train_path) / (1024 * 1024)
    print(f"\nwrote {train_path}  ({size_mb:.1f} MB)")

    report = {
        "sources": {"snomed": args.snomed, "icd": args.icd},
        "rows": {
            "snomed": snomed_r.num_rows,
            "icd":    icd_r.num_rows,
            "total":  final.num_rows,
        },
        "data_source_counts": dict(Counter(final.column("data_source").to_pylist())),
        "task_type_counts": dict(Counter(
            (ei or {}).get("task_type") for ei in final.column("extra_info").to_pylist()
        )),
        "output_train_parquet": train_path,
    }
    report_path = os.path.join(args.output, "_build_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {report_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
