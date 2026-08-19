"""Assemble the medical_qa STAGE-5 train + val parquets (coding focus).

Four pools, all-take (per user decision):
  ICD-ML       — `task_type="icd10_multilabel"`        →  9 828 train /  4 914 val
  ICD-IF       — `task_type="icd10_instruction_follow"`→  5 292 train /  1 059 val
  SNOMED-ML    — `task_type="snomed_multilabel"`       → 10 000 train /  5 000 val
  SNOMED-IF    — `task_type="snomed_instruction_follow"`→ 10 000 train /  2 400 val
                                                          ──────         ──────
                                                          35 120 train  13 373 val

Reconciliation per source:
  ICD (both):    convert `reward_model.ground_truth` from `list<string>` (or
                 `list<list<string>>` for IF) → JSON string.
  SNOMED (both): ground_truth is already a JSON string; just enforce the
                 canonical `struct<style: string, ground_truth: string>` shape.

extra_info is widened to the UNION of all four sources' fields + `task_type`
via `_coerce_struct_column`'s pylist round-trip. Top-level strings are
promoted to `large_string`. Output schema is identical for train and val.

Usage:
    python -m full_mix.curriculum.build_medical_qa_stage5 \\
        --output /data/abdelrahman/verl/data/medical_qa_stage5/

Both train.parquet and val.parquet are produced in one invocation (the four
source-pair paths are hard-coded for stage-5).
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
# Source pairs
# ============================================================================
_SOURCES = {
    "icd10_multilabel": {
        "train": "/home/mbinakba/mcs-instruct-dataset/avey_icd_multi_label_rl_train_dataset.parquet",
        "val":   "/home/mbinakba/mcs-instruct-dataset/avey_icd_multi_label_rl_test_dataset.parquet",
        "icd":   True,
    },
    "icd10_instruction_follow": {
        "train": "/home/mbinakba/mcs-instruct-dataset/avey_icd_if_rl_train_dataset.parquet",
        "val":   "/home/mbinakba/mcs-instruct-dataset/avey_icd_if_rl_test_dataset.parquet",
        "icd":   True,
    },
    "snomed_multilabel": {
        "train": "/data/muhsen/repos/finalize-dataset/abdelrahman_rl/final_ml_subsampled.parquet",
        "val":   "/data/muhsen/repos/finalize-dataset/abdelrahman_rl/final_ml_test.parquet",
        "icd":   False,
    },
    "snomed_instruction_follow": {
        "train": "/data/muhsen/repos/finalize-dataset/abdelrahman_rl/final_if_subsampled.parquet",
        "val":   "/data/muhsen/repos/finalize-dataset/abdelrahman_rl/final_if_test.parquet",
        "icd":   False,
    },
}


# ============================================================================
# Canonical types
# ============================================================================
_REWARD_MODEL_CANONICAL_TYPE = pa.struct([
    pa.field("style", pa.string()),
    pa.field("ground_truth", pa.string()),
])


# ============================================================================
# Per-pool helpers
# ============================================================================
def _drop_non_canonical_columns(t: pa.Table) -> pa.Table:
    """Keep only the 5 canonical top-level columns. Drops `code`, `description`,
    etc. that ship on some sources but aren't part of the verl loader contract."""
    canonical = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    return t.select([n for n in canonical if n in t.column_names])


def _stamp_task_type(table: pa.Table, task_type: str) -> pa.Table:
    """Add `extra_info.task_type = task_type` to every row, widening the
    struct type to include the new field if it isn't already there."""
    records = table.column("extra_info").to_pylist()
    for r in records:
        if isinstance(r, dict):
            r["task_type"] = task_type
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
    """ICD: convert `reward_model.ground_truth` from list<string> (ML) or
    list<list<string>> (IF) → JSON string. Mirrors the stage-4 build."""
    records = table.column("reward_model").to_pylist()
    new_rm = []
    for r in records:
        codes = r.get("ground_truth") if isinstance(r, dict) else None
        if codes is None:
            gt_str = "[]"
        elif isinstance(codes, str):
            gt_str = codes
        else:
            # list / list-of-list / numpy array → JSON via default=str fallback
            try:
                gt_str = json.dumps(list(codes), ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                gt_str = json.dumps(codes, ensure_ascii=False, default=str)
        new_rm.append({
            "style": (r.get("style") if isinstance(r, dict) else "") or "",
            "ground_truth": gt_str,
        })
    new_arr = pa.array(new_rm, type=_REWARD_MODEL_CANONICAL_TYPE)
    idx = table.column_names.index("reward_model")
    cols = list(table.columns)
    cols[idx] = new_arr
    return pa.table(cols, names=table.column_names)


def _snomed_normalise_reward_model(table: pa.Table) -> pa.Table:
    """SNOMED: ground_truth is already a JSON string. Just enforce the
    canonical struct type so all pools share one reward_model shape before concat."""
    records = table.column("reward_model").to_pylist()
    new_rm = []
    for r in records:
        gt = r.get("ground_truth") if isinstance(r, dict) else None
        if gt is None:
            gt_str = ""
        elif isinstance(gt, str):
            gt_str = gt
        else:
            try:
                gt_str = json.dumps(gt, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                gt_str = str(gt)
        new_rm.append({
            "style": (r.get("style") if isinstance(r, dict) else "") or "",
            "ground_truth": gt_str,
        })
    new_arr = pa.array(new_rm, type=_REWARD_MODEL_CANONICAL_TYPE)
    idx = table.column_names.index("reward_model")
    cols = list(table.columns)
    cols[idx] = new_arr
    return pa.table(cols, names=table.column_names)


# ============================================================================
# Schema reconciliation
# ============================================================================
def _build_unified_extra_info_type(types: list[pa.StructType]) -> pa.StructType:
    """Union of all field names + task_type. First occurrence wins on type."""
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


def _reconcile(t: pa.Table, unified_ei_type: pa.StructType) -> pa.Table:
    if t.schema.field("extra_info").type != unified_ei_type:
        t = _coerce_struct_column(t, "extra_info", unified_ei_type)
    if t.schema.field("reward_model").type != _REWARD_MODEL_CANONICAL_TYPE:
        t = _coerce_struct_column(t, "reward_model", _REWARD_MODEL_CANONICAL_TYPE)
    promoted = _promote_strings_to_large(t.schema)
    if promoted != t.schema:
        t = t.cast(promoted)
    return t


# ============================================================================
# Per-split build
# ============================================================================
def _build_split(split: str, output_dir: str) -> None:
    """Assemble one split (train | val) into `output_dir/{split}.parquet`."""
    print(f"\n========== building {split} ==========")
    pools: list[tuple[str, pa.Table]] = []
    for task_type, spec in _SOURCES.items():
        path = spec[split]
        t = pq.read_table(path)
        print(f"  {task_type:<28} {split:<5}  {path}  rows={t.num_rows}")
        t = _drop_non_canonical_columns(t)
        t = _stamp_task_type(t, task_type)
        t = (_icd_serialise_ground_truth if spec["icd"]
             else _snomed_normalise_reward_model)(t)
        pools.append((task_type, t))

    # Unified extra_info type
    ei_types = [t.schema.field("extra_info").type for _, t in pools]
    unified_ei = _build_unified_extra_info_type(ei_types)
    print(f"  unified extra_info: {unified_ei.num_fields} fields")
    print(f"  fields: {[f.name for f in unified_ei]}")

    # Reconcile + concat
    reconciled = [_reconcile(t, unified_ei) for _, t in pools]
    final = pa.concat_tables(reconciled, promote_options="default")
    print(f"  total {split} rows: {final.num_rows}")
    print(f"  data_source counts: {dict(Counter(final.column('data_source').to_pylist()))}")
    tt_counts = Counter(
        (ei or {}).get("task_type") for ei in final.column("extra_info").to_pylist()
    )
    print(f"  task_type counts  : {dict(tt_counts)}")

    # Sanity
    for r in final.column("reward_model").to_pylist()[:5]:
        assert isinstance(r["ground_truth"], str), \
            f"ground_truth must be str, got {type(r['ground_truth']).__name__}"
    for r in final.column("extra_info").to_pylist()[:5]:
        assert r.get("task_type") in _SOURCES, \
            f"task_type must be one of {sorted(_SOURCES)}, got {r.get('task_type')!r}"

    out_path = os.path.join(output_dir, f"{split}.parquet")
    pq.write_table(final, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  wrote {out_path}  ({size_mb:.1f} MB)")


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", required=True,
                    help="Output directory; writes train.parquet + val.parquet there.")
    args = ap.parse_args()

    print(f"[build_medical_qa_stage5] output_dir = {args.output}")
    print(f"  sources: {len(_SOURCES)} task types")
    for tt, spec in _SOURCES.items():
        print(f"    {tt:<28} icd={spec['icd']!s:<5}")

    os.makedirs(args.output, exist_ok=True)
    _build_split("train", args.output)
    _build_split("val",   args.output)

    # Confirm schema parity between train and val (verl's HF loader needs this).
    train = pq.read_table(os.path.join(args.output, "train.parquet"), columns=[
        "data_source", "extra_info", "reward_model",
    ])
    val   = pq.read_table(os.path.join(args.output, "val.parquet"), columns=[
        "data_source", "extra_info", "reward_model",
    ])
    print(f"\nschemas (train vs val):")
    print(f"  extra_info equal:  {train.schema.field('extra_info').type == val.schema.field('extra_info').type}")
    print(f"  reward_model equal:{train.schema.field('reward_model').type == val.schema.field('reward_model').type}")

    # Build report
    report = {
        "sources": {tt: spec for tt, spec in _SOURCES.items()},
        "train_rows": pq.read_metadata(os.path.join(args.output, "train.parquet")).num_rows,
        "val_rows":   pq.read_metadata(os.path.join(args.output, "val.parquet")).num_rows,
    }
    with open(os.path.join(args.output, "_build_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {os.path.join(args.output, '_build_report.json')}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
