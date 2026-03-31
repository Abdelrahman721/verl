# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Combine parquet outputs from snomed_preprocess.py (single-label) and
snomedml_preprocess.py (multilabel) into one training dataset.

Each row's reward_model.ground_truth JSON gains a "source" field:
  - "single"    -> single-label SNOMED (same schema as snomed_preprocess)
  - "multilabel" -> multilabel (same schema as snomedml_preprocess)

All rows use data_source == "snomed_mixed" so default_compute_score routes to
verl.utils.reward_score.snomed_mixed (per-entry dispatch to snomed vs sct_multilabel).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import datasets


MIXED_DATA_SOURCE = "snomed_mixed"


def _add_source_and_retag(
    example: Dict[str, Any],
    source: str,
    global_index: int,
) -> Dict[str, Any]:
    rm = dict(example["reward_model"])
    gt = json.loads(rm["ground_truth"])
    gt["source"] = source
    rm["ground_truth"] = json.dumps(gt, ensure_ascii=False)

    extra = dict(example.get("extra_info") or {})
    extra["index"] = global_index
    extra["mix_source"] = source

    return {
        **example,
        "data_source": MIXED_DATA_SOURCE,
        "reward_model": rm,
        "extra_info": extra,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge single-label and multilabel SNOMED parquets into one mixed dataset.",
    )
    parser.add_argument(
        "--single_parquet",
        type=str,
        required=True,
        help="Parquet from snomed_preprocess.py (single-label CSV -> parquet).",
    )
    parser.add_argument(
        "--multilabel_parquet",
        type=str,
        required=True,
        help="Parquet from snomedml_preprocess.py (e.g. multilabel_combined_train_rl.parquet).",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/workspace/verl/rl-data/",
        help="Directory to save the combined parquet.",
    )
    parser.add_argument(
        "--out_parquet_name",
        default="snomed_mixed_train.parquet",
        help="Output parquet filename (written under local_save_dir).",
    )
    parser.add_argument(
        "--single_first",
        action="store_true",
        help="If set, concatenate single-label rows before multilabel (default: multilabel first).",
    )
    args = parser.parse_args()

    single_path = Path(args.single_parquet)
    ml_path = Path(args.multilabel_parquet)
    if not single_path.is_file():
        raise FileNotFoundError(f"Single-label parquet not found: {single_path}")
    if not ml_path.is_file():
        raise FileNotFoundError(f"Multilabel parquet not found: {ml_path}")

    ds_single = datasets.load_dataset("parquet", data_files=str(single_path), split="train")
    ds_ml = datasets.load_dataset("parquet", data_files=str(ml_path), split="train")

    rows: list[Dict[str, Any]] = []
    idx = 0

    order: list[tuple[datasets.Dataset, str]] = (
        [(ds_single, "single"), (ds_ml, "multilabel")]
        if args.single_first
        else [(ds_ml, "multilabel"), (ds_single, "single")]
    )
    for ds, source_key in order:
        for ex in ds:
            rows.append(_add_source_and_retag(ex, source_key, idx))
            idx += 1

    combined = datasets.Dataset.from_list(rows)
    save_dir = Path(os.path.expanduser(args.local_save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / args.out_parquet_name
    combined.to_parquet(str(out_path))

    print(f"Saved combined parquet to: {out_path}")
    print(f"Total examples: {len(combined)} ({len(ds_single)} single + {len(ds_ml)} multilabel)")
    print(f"data_source for all rows: {MIXED_DATA_SOURCE!r}")
