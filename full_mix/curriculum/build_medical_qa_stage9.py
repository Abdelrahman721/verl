"""Assemble the medical_qa STAGE-9 train + val parquets (real coding, large retention).

Same train sources as stage 8 with different per-source epoch counts.
The stage-9 trainer DOES consume retention.parquet (vs stage 8's
no-retention baseline). Build script only handles train + val here;
retention is built separately by `build_medical_qa_stage9_retention.py`.

Two sources, both multilabel scoring, per-source in-parquet duplication for
epoch weighting:

  mlb         — avey_mlb_train_dataset.parquet (308 × 3 →  924)
                ICD-10 multilabel; comma-separated string gold.
  snomed_ml   — final_ml_real.parquet          (103 × 8 →  824)
                SNOMED CT multilabel; JSON-string gold (gt_sct_ids list).
                                                       ─────
                                                       1 748 train

Val (no duplication):
  mlb test         77
  snomed_ml test   68
                 ───
                 145 val

Stamping:
  mlb:        task_type = "icd10_multilabel"
              ground_truth (csv string) → JSON list of code strings
  snomed_ml:  task_type = "snomed_multilabel"
              ground_truth pass-through (already JSON string)

Schema harmonisation: extra_info widened to the union of both sources'
fields + `task_type`; prompt struct field order normalised to (role, content);
top-level strings promoted to large_string. Identical schema in train and val.

Usage:
    python -m full_mix.curriculum.build_medical_qa_stage9 \\
        --output /data/abdelrahman/verl/data/medical_qa_stage9/
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
# Sources
# ============================================================================
# Each entry:
#   train / val parquet paths
#   data_source (matches what's already in the parquet — not rewritten)
#   task_type to stamp on extra_info
#   epochs: in-parquet duplication factor for train
#   gold_kind: "csv_string" | "list_like" | "json_string_passthrough"
_SOURCES = {
    "mlb": {
        "train":     "/home/mbinakba/mcs-instruct-dataset/avey_mlb_train_dataset.parquet",
        "val":       "/home/mbinakba/mcs-instruct-dataset/avey_mlb_test_dataset.parquet",
        "task_type": "icd10_multilabel",
        "epochs":    3,
        "gold_kind": "csv_string",
    },
    "snomed_ml": {
        "train":     "/data/muhsen/repos/finalize-dataset/abdelrahman_rl/final_ml_real.parquet",
        "val":       "/data/muhsen/repos/finalize-dataset/abdelrahman_rl/final_ml_real_test.parquet",
        "task_type": "snomed_multilabel",
        "epochs":    8,
        "gold_kind": "json_string_passthrough",
    },
}


# ============================================================================
# Canonical types
# ============================================================================
_REWARD_MODEL_CANONICAL_TYPE = pa.struct([
    pa.field("style", pa.string()),
    pa.field("ground_truth", pa.string()),
])

_PROMPT_CANONICAL_TYPE = pa.list_(pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
]))


# ============================================================================
# Per-pool helpers
# ============================================================================
def _drop_non_canonical_columns(t: pa.Table) -> pa.Table:
    canonical = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    return t.select([n for n in canonical if n in t.column_names])


# Fields that collide with stage-5 retention's extra_info shape — dropped
# from stage-6 train's extra_info because they're pure telemetry (redundant
# with reward_model.ground_truth) and their type mismatch (`codes` is a
# csv string here vs list<string> there; `code` is list<string> here vs
# list<list<string>> there) blocks schema unification with retention.parquet.
_DROP_EXTRA_INFO_FIELDS_ON_STAMP = frozenset({"codes", "code"})


def _stamp_task_type(table: pa.Table, task_type: str) -> pa.Table:
    """Add `extra_info.task_type = task_type` to every row, widening the
    struct type to include the new field if it isn't already there. Also
    drops any fields in `_DROP_EXTRA_INFO_FIELDS_ON_STAMP` to avoid retention-
    side schema collisions."""
    records = table.column("extra_info").to_pylist()
    for r in records:
        if isinstance(r, dict):
            for k in _DROP_EXTRA_INFO_FIELDS_ON_STAMP:
                r.pop(k, None)
            r["task_type"] = task_type
    cur_type = table.schema.field("extra_info").type
    kept_fields = [f for f in cur_type
                   if f.name not in _DROP_EXTRA_INFO_FIELDS_ON_STAMP]
    if not any(f.name == "task_type" for f in kept_fields):
        kept_fields.append(pa.field("task_type", pa.string()))
    augmented = pa.struct(kept_fields)
    new_arr = pa.array(records, type=augmented)
    idx = table.column_names.index("extra_info")
    cols = list(table.columns)
    cols[idx] = new_arr
    return pa.table(cols, names=table.column_names)


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


def _normalise_ground_truth(table: pa.Table, gold_kind: str) -> pa.Table:
    """Coerce reward_model.ground_truth to canonical JSON-string shape."""
    records = table.column("reward_model").to_pylist()
    new_rm = []
    for r in records:
        gt = r.get("ground_truth") if isinstance(r, dict) else None
        if gold_kind == "csv_string":
            # "D63.8, E04.1, E11.22, ..." → JSON list of codes
            if gt is None:
                codes = []
            elif isinstance(gt, str):
                codes = [c.strip() for c in gt.split(",") if c.strip()]
            elif isinstance(gt, (list, tuple)):
                codes = [str(c).strip() for c in gt if str(c).strip()]
            else:
                # numpy array etc.
                codes = [str(c).strip() for c in list(gt) if str(c).strip()]
            gt_str = json.dumps(codes, ensure_ascii=False, default=str)
        elif gold_kind == "list_like":
            # ndarray of code strings → JSON list
            if gt is None:
                codes = []
            elif isinstance(gt, str):
                # Already a string — try to detect JSON shape, else wrap single code
                stripped = gt.strip()
                if stripped.startswith("["):
                    gt_str = stripped
                    new_rm.append({"style": (r.get("style") or "") if isinstance(r, dict) else "",
                                   "ground_truth": gt_str})
                    continue
                codes = [stripped] if stripped else []
            else:
                codes = [str(c).strip() for c in list(gt) if str(c).strip()]
            gt_str = json.dumps(codes, ensure_ascii=False, default=str)
        else:  # json_string_passthrough
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
            "style": (r.get("style") or "") if isinstance(r, dict) else "",
            "ground_truth": gt_str,
        })
    new_arr = pa.array(new_rm, type=_REWARD_MODEL_CANONICAL_TYPE)
    idx = table.column_names.index("reward_model")
    cols = list(table.columns)
    cols[idx] = new_arr
    return pa.table(cols, names=table.column_names)


def _duplicate_table(t: pa.Table, n: int) -> pa.Table:
    """Concat the table with itself N times. n=1 is a no-op."""
    if n <= 1:
        return t
    return pa.concat_tables([t] * n)


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
def _build_split(split: str, output_dir: str) -> int:
    print(f"\n========== building {split} ==========")
    pools: list[tuple[str, pa.Table]] = []
    for src, spec in _SOURCES.items():
        path = spec[split]
        t = pq.read_table(path)
        n_in = t.num_rows
        t = _drop_non_canonical_columns(t)
        t = _normalise_prompt_struct(t)
        t = _stamp_task_type(t, spec["task_type"])
        t = _normalise_ground_truth(t, spec["gold_kind"])
        # Train-only: duplicate per epoch
        if split == "train" and spec["epochs"] > 1:
            t = _duplicate_table(t, spec["epochs"])
            print(f"  {src:<10} {split:<5} rows={n_in:>5} × {spec['epochs']} epochs = {t.num_rows:>5}  ({path})")
        else:
            print(f"  {src:<10} {split:<5} rows={n_in:>5}                       ({path})")
        pools.append((src, t))

    ei_types = [t.schema.field("extra_info").type for _, t in pools]
    unified_ei = _build_unified_extra_info_type(ei_types)
    print(f"  unified extra_info: {unified_ei.num_fields} fields")
    print(f"  fields: {[f.name for f in unified_ei]}")

    reconciled = [_reconcile(t, unified_ei) for _, t in pools]
    final = pa.concat_tables(reconciled, promote_options="default")
    print(f"  total {split} rows: {final.num_rows}")
    print(f"  data_source: {dict(Counter(final.column('data_source').to_pylist()))}")
    tt_counts = Counter(
        (ei or {}).get("task_type") for ei in final.column("extra_info").to_pylist()
    )
    print(f"  task_type:   {dict(tt_counts)}")

    # Sanity
    for r in final.column("reward_model").to_pylist()[:3]:
        assert isinstance(r["ground_truth"], str), \
            f"ground_truth must be str, got {type(r['ground_truth']).__name__}"
    for r in final.column("extra_info").to_pylist()[:3]:
        assert r.get("task_type") in {"icd10_multilabel", "snomed_multilabel"}, \
            f"unexpected task_type: {r.get('task_type')!r}"

    out_path = os.path.join(output_dir, f"{split}.parquet")
    pq.write_table(final, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  wrote {out_path}  ({size_mb:.1f} MB)")
    return final.num_rows


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

    print(f"[build_medical_qa_stage6] output_dir = {args.output}")
    for src, spec in _SOURCES.items():
        print(f"  {src:<10} task_type={spec['task_type']:<18} "
              f"epochs={spec['epochs']}  gold={spec['gold_kind']}")

    os.makedirs(args.output, exist_ok=True)
    n_train = _build_split("train", args.output)
    n_val   = _build_split("val",   args.output)

    # Schema parity check
    train = pq.read_table(os.path.join(args.output, "train.parquet"))
    val   = pq.read_table(os.path.join(args.output, "val.parquet"))
    ei_eq  = train.schema.field("extra_info").type == val.schema.field("extra_info").type
    rm_eq  = train.schema.field("reward_model").type == val.schema.field("reward_model").type
    print(f"\nschema parity (train vs val):  extra_info={ei_eq}  reward_model={rm_eq}")
    if not (ei_eq and rm_eq):
        # Same fix as stage-5: widen val's extra_info to train's, or vice versa.
        # If they differ here, it's a build bug — emit a clear failure.
        print("  WARNING: train/val schemas diverge — concat may fail downstream.")

    report = {
        "sources":    {k: v for k, v in _SOURCES.items()},
        "train_rows": n_train,
        "val_rows":   n_val,
    }
    with open(os.path.join(args.output, "_build_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {os.path.join(args.output, '_build_report.json')}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
