"""Harmonise the Arrow schemas of several parquets that verl loads as ONE dataset.

Stages 5 and 6 hand verl two train parquets (``train.parquet`` +
``retention.parquet``). verl loads them through HF ``datasets``, which
concat-loads the files into a single dataset and therefore requires every file
to carry an IDENTICAL Arrow schema. The per-stage builders each unify schemas
*within* their own output, but nothing reconciles a coding ``train.parquet``
against a separately-built ``retention.parquet`` — in the original 32B run this
was a manual pyarrow pass, evidenced only by the ``*.preharmonise.bak`` files
left beside the outputs.

This script is that pass, made reproducible. Given N parquets it:

  1. drops non-canonical top-level columns (keeps data_source / prompt /
     ability / reward_model / extra_info),
  2. normalises ``prompt`` to list<struct<role, content>>,
  3. normalises ``reward_model`` to struct<style, ground_truth>,
  4. widens ``extra_info`` to the UNION of every file's fields (first
     occurrence wins on type, ``task_type`` always present),
  5. promotes string → large_string so the files cast cleanly,
  6. rewrites each file in place, after copying the original to
     ``<name>.preharmonise.bak`` unless --no-backup is passed.

Idempotent: running it twice is a no-op the second time (schemas already match).

Usage:
    python -m qlcm.curriculum.harmonise_train_files \\
        data/qlcm/medical_qa_stage5/train.parquet \\
        data/qlcm/medical_qa_stage5/retention.parquet

Field-type conflicts (same name, different type across files) are FATAL by
default — silently keeping the first type would corrupt the other file's rows.
Pass --drop-conflicts to drop the offending fields from extra_info instead.
"""

import argparse
import json
import os
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_QLCM_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_QLCM_DIR)
for _p in (_REPO_ROOT, _QLCM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qlcm.curriculum.build_curated_subset import _promote_strings_to_large  # noqa: E402
from qlcm.curriculum.build_medical_qa_next import _coerce_struct_column  # noqa: E402


_CANONICAL_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

_REWARD_MODEL_CANONICAL_TYPE = pa.struct([
    pa.field("style", pa.string()),
    pa.field("ground_truth", pa.string()),
])

_PROMPT_CANONICAL_TYPE = pa.list_(pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
]))


def _drop_non_canonical_columns(t: pa.Table) -> pa.Table:
    return t.select([n for n in _CANONICAL_COLUMNS if n in t.column_names])


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


def _unified_extra_info_type(types, drop_conflicts: bool):
    """Union of every file's extra_info fields. First occurrence wins on type.

    Returns (struct_type, conflicts) where conflicts maps field name -> the
    list of distinct types seen for it.
    """
    seen = {}
    conflicts = {}
    for t in types:
        for i in range(t.num_fields):
            f = t.field(i)
            if f.name not in seen:
                seen[f.name] = f
            elif seen[f.name].type != f.type:
                conflicts.setdefault(f.name, [str(seen[f.name].type)]).append(str(f.type))

    if conflicts and drop_conflicts:
        for name in conflicts:
            seen.pop(name, None)

    fields = list(seen.values())
    if "task_type" not in seen:
        fields.append(pa.field("task_type", pa.string()))
    return pa.struct(fields), conflicts


def _reconcile(t: pa.Table, unified_ei_type: pa.StructType) -> pa.Table:
    t = _drop_non_canonical_columns(t)
    t = _normalise_prompt_struct(t)
    if t.schema.field("extra_info").type != unified_ei_type:
        t = _coerce_struct_column(t, "extra_info", unified_ei_type)
    if t.schema.field("reward_model").type != _REWARD_MODEL_CANONICAL_TYPE:
        t = _coerce_struct_column(t, "reward_model", _REWARD_MODEL_CANONICAL_TYPE)
    promoted = _promote_strings_to_large(t.schema)
    if promoted != t.schema:
        t = t.cast(promoted)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="+",
                    help="Two or more parquets that verl will load as one dataset.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip writing <name>.preharmonise.bak.")
    ap.add_argument("--drop-conflicts", action="store_true",
                    help="Drop extra_info fields whose type differs across files "
                         "instead of failing.")
    ap.add_argument("--report", default="",
                    help="Optional path for a JSON report of what was unified.")
    args = ap.parse_args()

    if len(args.parquets) < 2:
        ap.error("give at least two parquets — harmonising one file is a no-op")
    for p in args.parquets:
        if not os.path.exists(p):
            ap.error(f"missing parquet: {p}")

    tables = []
    for p in args.parquets:
        t = pq.read_table(p)
        missing = [c for c in ("prompt", "reward_model", "extra_info") if c not in t.column_names]
        if missing:
            ap.error(f"{p} is missing required column(s): {missing}")
        print(f"  read  {p}  rows={t.num_rows}  extra_info fields="
              f"{t.schema.field('extra_info').type.num_fields}")
        tables.append(t)

    ei_types = [t.schema.field("extra_info").type for t in tables]
    unified, conflicts = _unified_extra_info_type(ei_types, args.drop_conflicts)

    if conflicts and not args.drop_conflicts:
        print("\nFATAL: extra_info field-type conflicts across files:", file=sys.stderr)
        for name, types in conflicts.items():
            print(f"  {name}: {' vs '.join(sorted(set(types)))}", file=sys.stderr)
        print("\nRe-run with --drop-conflicts to drop these fields, or fix the "
              "builders so they agree on a type.", file=sys.stderr)
        return 2
    if conflicts:
        print(f"\n  dropped {len(conflicts)} conflicting extra_info field(s): "
              f"{sorted(conflicts)}")

    print(f"\n  unified extra_info: {unified.num_fields} fields")

    written = []
    for path, t in zip(args.parquets, tables):
        out = _reconcile(t, unified)
        if out.schema.equals(t.schema):
            print(f"  skip  {path}  (already harmonised)")
            written.append({"path": path, "rows": out.num_rows, "rewritten": False})
            continue
        if not args.no_backup:
            bak = path + ".preharmonise.bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
                print(f"  backup {bak}")
        pq.write_table(out, path)
        print(f"  write {path}  rows={out.num_rows}")
        written.append({"path": path, "rows": out.num_rows, "rewritten": True})

    # Every file must now agree, or verl's concat-load will still fail.
    schemas = [pq.read_schema(p) for p in args.parquets]
    if not all(s.equals(schemas[0]) for s in schemas[1:]):
        print("\nFATAL: schemas still differ after harmonisation.", file=sys.stderr)
        for p, s in zip(args.parquets, schemas):
            print(f"\n--- {p}\n{s}", file=sys.stderr)
        return 3
    print("\n  OK — all files share one schema")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump({
                "inputs": args.parquets,
                "unified_extra_info_fields": [f.name for f in unified],
                "dropped_conflicting_fields": sorted(conflicts),
                "files": written,
            }, fh, indent=2)
        print(f"  report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
