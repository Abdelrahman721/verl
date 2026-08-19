"""Verify that ``medical_qa_compat`` is lossless: for every row in the source
parquet, ``json.loads(transformed.reward_model.ground_truth)`` is deep-equal to
the original ``reward_model.ground_truth`` struct.

This is the exact code path qa_bedrock takes on every call (the
``if isinstance(ground_truth, str): ground_truth = json.loads(ground_truth)``
branch at the top of ``_compute_score_single``), so a row-wise pass here
proves the transform is a no-op from qa_bedrock's point of view.

Usage:
    # Defaults: verify val.parquet (48 rows, ~1s)
    python -m full_mix.preprocess.verify_medical_qa_compat

    # Verify train.parquet end-to-end (316,287 rows, ~30s)
    python -m full_mix.preprocess.verify_medical_qa_compat --files train.parquet

    # Both:
    python -m full_mix.preprocess.verify_medical_qa_compat \\
        --files train.parquet val.parquet
"""

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

from full_mix.preprocess.medical_qa_compat import transform_table


def deep_equal(a, b) -> bool:
    """Structural equality that's strict about types AND key sets, including
    NaN-aware float compare and ordered list compare.

    PyArrow's ``to_pylist()`` materializes nested structs as plain Python
    dicts/lists, so this works on both the pre- and post-transform sides.
    """
    if type(a) is not type(b):
        # Permit int/float mixing only if they're numerically equal — but
        # arrow shouldn't be doing that for us, so we want to know if it does.
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        # NaNs are never equal in Python; treat them as equal here so
        # genuine NaN ground_truth values don't trip the check.
        if a != a and b != b:
            return True
    return a == b


def verify_file(src_path: str) -> tuple[int, int, list[tuple[int, str]]]:
    """Returns (rows_checked, rows_passed, list_of_failures)."""
    print(f"\n[verifying {os.path.basename(src_path)}]")
    src_table = pq.read_table(src_path)
    n = src_table.num_rows
    print(f"  rows = {n}")

    src_rm = src_table.column("reward_model").to_pylist()

    # Run the actual transform used by medical_qa_compat, then re-read
    # the transformed ground_truth back into Python via json.loads — i.e.
    # exactly what qa_bedrock will do at inference time.
    dst_table = transform_table(src_table)
    dst_rm = dst_table.column("reward_model").to_pylist()

    failures: list[tuple[int, str]] = []
    passed = 0

    for i in range(n):
        orig_gt = src_rm[i]["ground_truth"]            # dict (struct)
        dst_gt_str = dst_rm[i]["ground_truth"]         # JSON string

        if not isinstance(dst_gt_str, str):
            failures.append((i, f"post-transform ground_truth is {type(dst_gt_str).__name__}, expected str"))
            continue

        try:
            recovered = json.loads(dst_gt_str)
        except json.JSONDecodeError as e:
            failures.append((i, f"json.loads raised JSONDecodeError: {e}"))
            continue

        # Both should be plain Python dicts (or None) at this point.
        if not deep_equal(orig_gt, recovered):
            # Find the first diverging key for a useful message.
            divergence = _first_divergence(orig_gt, recovered, path="ground_truth")
            failures.append((i, divergence))
            continue

        # Also verify style is preserved exactly.
        if src_rm[i].get("style") != dst_rm[i].get("style"):
            failures.append((i, f"style drift: {src_rm[i].get('style')!r} -> {dst_rm[i].get('style')!r}"))
            continue

        passed += 1

    return n, passed, failures


def _first_divergence(a, b, path: str) -> str:
    """Return a short string pinpointing the first place a and b differ."""
    if type(a) is not type(b):
        return f"{path}: type mismatch {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        ka, kb = set(a.keys()), set(b.keys())
        if ka != kb:
            return f"{path}: key set mismatch (only_a={ka - kb!r}, only_b={kb - ka!r})"
        for k in a:
            if not deep_equal(a[k], b[k]):
                return _first_divergence(a[k], b[k], path=f"{path}.{k}")
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: list length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if not deep_equal(x, y):
                return _first_divergence(x, y, path=f"{path}[{i}]")
    return f"{path}: value mismatch ({a!r} vs {b!r})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir",
        default="/data/abdelrahman/verl/data/medical_qa",
        help="Where the source medical_qa parquets live.",
    )
    ap.add_argument(
        "--files",
        nargs="+",
        default=["val.parquet"],
        help="Which files to verify. Default: val.parquet (fast). "
             "Pass train.parquet for the full sweep (~30s).",
    )
    args = ap.parse_args()

    total = 0
    total_passed = 0
    all_failures: list[tuple[str, int, str]] = []
    for fn in args.files:
        path = os.path.join(args.src_dir, fn)
        if not os.path.exists(path):
            print(f"\n[{fn}] SKIP — not found at {path}")
            continue
        n, passed, fails = verify_file(path)
        total += n
        total_passed += passed
        for idx, msg in fails:
            all_failures.append((fn, idx, msg))
        print(f"  passed = {passed:>6} / {n:>6}   ({100 * passed / n:.2f}%)")
        if fails:
            print(f"  failed = {len(fails)} rows (first 5 listed):")
            for fn_, idx, msg in fails[:5]:
                print(f"    [{fn_}#{idx}] {msg}")

    print("\n" + "=" * 70)
    if not all_failures:
        print(f"ALL PASS — {total_passed}/{total} rows round-trip losslessly")
        return 0
    else:
        print(f"FAILURES — {len(all_failures)} rows out of {total} did NOT round-trip")
        return 1


if __name__ == "__main__":
    sys.exit(main())
