"""Cross-source verification.

For every row i:

    local_ground_truth_str = local_parquet[i].reward_model.ground_truth
    recovered_struct       = json.loads(local_ground_truth_str)
    source_struct          = source_parquet[i].reward_model.ground_truth   # from hazem's untouched parquet
    assert deep_equal(recovered_struct, source_struct)

This is the strictest possible check that ``medical_qa_compat`` produced a
lossless on-disk representation of hazem's source data:

  - The local file (already rewritten by ``medical_qa_compat`` in place) is
    expected to have ``reward_model.ground_truth`` as a JSON STRING.
  - Hazem's source file (untouched, READ-ONLY) is expected to have
    ``reward_model.ground_truth`` as a nested STRUCT.
  - We json.loads the local string, then deep-equal it against the source struct
    on a per-row basis.

This script does NOT write to either file. It opens both for read.

Usage:
    # Default: verify val.parquet (48 rows, ~1s)
    python -m full_mix.preprocess.verify_local_against_source

    # Verify train (316,287 rows, ~30s)
    python -m full_mix.preprocess.verify_local_against_source \\
        --files train.parquet val.parquet
"""

import argparse
import hashlib
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from full_mix.preprocess.verify_medical_qa_compat import (
    deep_equal,
    _first_divergence,
)


DEFAULT_LOCAL  = "/data/abdelrahman/verl/data/medical_qa"
DEFAULT_SOURCE = "/data/hazem/medical-data-gen/processed/rl/combined/v1"


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pair(local_path: str, source_path: str) -> tuple[int, int, list[tuple[int, str]]]:
    """Returns (rows_checked, rows_passed, list_of_failures)."""
    print(f"\n[verifying {os.path.basename(local_path)}]")

    # Tamper-evidence on the source side: record md5 before AND after
    # to prove this script doesn't touch hazem's file.
    source_md5_before = _md5(source_path)

    local_table  = pq.read_table(local_path)
    source_table = pq.read_table(source_path)
    n_local  = local_table.num_rows
    n_source = source_table.num_rows
    if n_local != n_source:
        print(f"  row-count mismatch: local={n_local} source={n_source} — cannot pair")
        return n_local, 0, [(-1, f"row count mismatch local={n_local} source={n_source}")]
    n = n_local
    print(f"  rows = {n}")
    print(f"  local  : {local_path}")
    print(f"  source : {source_path}")

    # Sanity: confirm the local rewards column actually IS a string column,
    # i.e. that medical_qa_compat has been run on it. If not, abort with a
    # clear message so the user knows what to do.
    local_rm_field = local_table.schema.field("reward_model").type
    if not pa.types.is_string(local_rm_field.field("ground_truth").type):
        print(f"  ERROR: local reward_model.ground_truth is {local_rm_field.field('ground_truth').type}, "
              f"expected string. Run `python -m full_mix.preprocess.medical_qa_compat` first.")
        return n, 0, [(-1, "local ground_truth is not a string — medical_qa_compat hasn't been run on this file")]

    # Materialise both reward_model columns to Python and walk them in parallel.
    local_rm  = local_table.column("reward_model").to_pylist()
    source_rm = source_table.column("reward_model").to_pylist()

    # Independently double-check the per-row data_source column matches, to
    # rule out any row-order divergence between the two files (medical_qa_compat
    # preserves order, but this is an extra belt).
    local_ds  = local_table.column("data_source").to_pylist()
    source_ds = source_table.column("data_source").to_pylist()

    failures: list[tuple[int, str]] = []
    passed = 0

    for i in range(n):
        if local_ds[i] != source_ds[i]:
            failures.append((i, f"data_source order drift: local={local_ds[i]!r} source={source_ds[i]!r}"))
            continue

        local_gt_str = local_rm[i].get("ground_truth")
        if not isinstance(local_gt_str, str):
            failures.append((i, f"local ground_truth is {type(local_gt_str).__name__}, expected str"))
            continue

        try:
            recovered_struct = json.loads(local_gt_str)
        except json.JSONDecodeError as e:
            failures.append((i, f"json.loads(local) raised JSONDecodeError: {e}"))
            continue

        source_struct = source_rm[i].get("ground_truth")
        if not isinstance(source_struct, dict):
            failures.append((i, f"source ground_truth is {type(source_struct).__name__}, expected dict (struct)"))
            continue

        if not deep_equal(recovered_struct, source_struct):
            div = _first_divergence(recovered_struct, source_struct, path="ground_truth")
            failures.append((i, div))
            continue

        if local_rm[i].get("style") != source_rm[i].get("style"):
            failures.append((i, f"style drift: local={local_rm[i].get('style')!r} "
                                f"source={source_rm[i].get('style')!r}"))
            continue

        passed += 1

    # Tamper-evidence: source md5 must be unchanged after the read.
    source_md5_after = _md5(source_path)
    if source_md5_before != source_md5_after:
        print(f"  ⚠️  source md5 changed during verify (before={source_md5_before} "
              f"after={source_md5_after}) — something tampered with the source file")
    else:
        print(f"  source md5 unchanged: {source_md5_before}")

    return n, passed, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir",  default=DEFAULT_LOCAL,
                    help="Directory containing the LOCAL (post-medical_qa_compat) parquets.")
    ap.add_argument("--source_dir", default=DEFAULT_SOURCE,
                    help="Directory containing hazem's UNTOUCHED source parquets. READ ONLY.")
    ap.add_argument("--files", nargs="+", default=["val.parquet"],
                    help="Filenames to compare (must exist in BOTH directories). Default: val.parquet.")
    args = ap.parse_args()

    total = 0
    total_passed = 0
    any_failure = False

    for fn in args.files:
        local_path  = os.path.join(args.local_dir,  fn)
        source_path = os.path.join(args.source_dir, fn)
        if not os.path.exists(local_path):
            print(f"\n[{fn}] SKIP — local missing: {local_path}")
            any_failure = True
            continue
        if not os.path.exists(source_path):
            print(f"\n[{fn}] SKIP — source missing: {source_path}")
            any_failure = True
            continue

        n, passed, fails = verify_pair(local_path, source_path)
        total += n
        total_passed += passed
        print(f"  passed = {passed:>6} / {n:>6}   ({100 * passed / n:.2f}%)" if n else
              f"  passed = 0 / 0 (nothing to check)")
        if fails:
            any_failure = True
            print(f"  failed = {len(fails)} rows (first 5 listed):")
            for idx, msg in fails[:5]:
                tag = f"#{idx}" if idx >= 0 else "(file-level)"
                print(f"    [{fn}{tag}] {msg}")

    print("\n" + "=" * 70)
    if not any_failure:
        print(f"ALL PASS — every row's local string ground_truth json.loads back to "
              f"hazem's source struct ({total_passed}/{total} rows checked).")
        return 0
    print(f"FAILURES — see above. {total_passed}/{total} rows passed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
