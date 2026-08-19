"""Re-emit the medical_qa parquets with ``reward_model.ground_truth`` serialized
to a JSON string so they can be concatenated with the full_mix retention
parquets at training time.

Background
----------
The medical_qa parquets from hazem's pipeline ship with ``ground_truth`` as a
nested struct (``gold_answer``, ``key_points`` list-of-structs, ``gold_response``,
``context``, ``latest_user``, ``type_descriptor``, ``sub_index``, ``total_subs``).
The full_mix parquets (ifeval / chat / safety / identity — including the
curated retention subsets) use a plain ``string`` ground_truth.

HuggingFace Datasets refuses to concatenate the two schemas:

    ValueError: The features can't be aligned because the key ground_truth of
    features {'ground_truth': Value('string'), 'style': Value('string')} has
    unexpected type - Value('string') (expected either {'gold_answer': ...} ...

Why this is safe
----------------
``qa_bedrock.compute_score`` already accepts a JSON-string ground_truth — see
the ``if isinstance(ground_truth, str): ground_truth = json.loads(ground_truth)``
branch at the top of ``_compute_score_single``. So serializing the struct does
NOT require any reward-function changes; the qa_bedrock contract is preserved
verbatim.

All other columns (``prompt``, ``data_source``, ``ability``, ``extra_info``) are
passed through unchanged. The ``large_string`` Arrow types on ``data_source`` and
``ability`` are preserved.

Usage
-----
    # Overwrite in place (default — recommended; original is still backed up
    # at /data/hazem/medical-data-gen/processed/rl/combined/v1/):
    python -m full_mix.preprocess.medical_qa_compat

    # Write to a separate directory (non-destructive):
    python -m full_mix.preprocess.medical_qa_compat \\
        --dst_dir /data/abdelrahman/verl/data/medical_qa_compat
"""

import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def transform_table(t: pa.Table) -> pa.Table:
    """Replace ``reward_model.ground_truth`` (struct) with a JSON string.

    Everything else (column order, top-level types — including
    ``large_string`` on ``data_source``/``ability``, ``extra_info`` struct, etc.)
    is preserved bit-for-bit.
    """
    rm_records = t.column("reward_model").to_pylist()
    new_rm = [
        {
            "style": r.get("style", ""),
            "ground_truth": json.dumps(r.get("ground_truth", {}), ensure_ascii=False),
        }
        for r in rm_records
    ]
    rm_type = pa.struct(
        [
            pa.field("style", pa.string()),
            pa.field("ground_truth", pa.string()),
        ]
    )
    new_rm_array = pa.array(new_rm, type=rm_type)

    new_columns = {}
    for name in t.column_names:
        new_columns[name] = new_rm_array if name == "reward_model" else t.column(name)
    return pa.table(new_columns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir",
        default="/data/abdelrahman/verl/data/medical_qa",
        help="Directory containing the medical_qa parquets.",
    )
    ap.add_argument(
        "--dst_dir",
        default=None,
        help="Where to write the rewritten parquets. Defaults to --src_dir (overwrite in place).",
    )
    ap.add_argument(
        "--files",
        nargs="+",
        default=["train.parquet", "val.parquet"],
        help="Parquet filenames to transform (relative to --src_dir).",
    )
    args = ap.parse_args()

    dst_dir = args.dst_dir or args.src_dir
    os.makedirs(dst_dir, exist_ok=True)
    inplace = os.path.realpath(dst_dir) == os.path.realpath(args.src_dir)

    print(f"src_dir = {args.src_dir}")
    print(f"dst_dir = {dst_dir}  ({'OVERWRITE in place' if inplace else 'separate'})")

    for fn in args.files:
        src = os.path.join(args.src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        if not os.path.exists(src):
            print(f"\n[{fn}] SKIP — source not found: {src}")
            continue

        print(f"\n[{fn}] reading {src} …")
        t = pq.read_table(src)
        n = t.num_rows
        before = t.schema.field("reward_model").type

        t2 = transform_table(t)
        after = t2.schema.field("reward_model").type
        # Sanity: row count unchanged, schema matches expectations
        assert t2.num_rows == n, f"row count drift: {n} -> {t2.num_rows}"
        assert pa.types.is_string(after.field("ground_truth").type), (
            f"unexpected ground_truth type after transform: {after}"
        )

        print(f"  rows                        : {n}")
        print(f"  reward_model BEFORE         : {before}")
        print(f"  reward_model AFTER          : {after}")
        pq.write_table(t2, dst)
        print(f"  wrote -> {dst}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
