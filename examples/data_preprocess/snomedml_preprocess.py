# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Preprocess multilabel SNOMED CSV to parquet format for VERL RLHF.

This script combines all multilabel CSVs under:
  /data/muhsen/repos/verl/Datasets/train_multilabel
into a single parquet dataset, while:
  - parsing the JSON-encoded multilabel `sct_id` column
  - adding the model-style prefix to each example id (cla_/gem_/gpt_)
  - emitting the same general parquet structure as the original preprocess script.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets


SYSTEM_PROMPT = (
    # Must match `llm-pretrainer/convert_mlsnomed_to_sft.py`
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

USER_PROMPT_TEMPLATE = (
    # Must match `llm-pretrainer/convert_mlsnomed_to_sft.py`
    "You are a SNOMED CT medical coding expert. You are given a clinical note and you need to assign all applicable SNOMED CT concepts.\n"
    "Think step by step and provide all applicable codes with their official descriptions.\n\n"
    "Clinical note:\n{entry}"
)


def _prefix_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "claude" in name:
        return "cla_"
    if "gemini" in name:
        return "gem_"
    if "gpt" in name:
        return "gpt_"
    return "src_"


def _parse_multilabel_sct_id(cell: str) -> Tuple[List[str], List[str]]:
    """
    Parse CSV `sct_id` shaped like:
      [["id1","label1"],["id2","label2"],...]
    Returns (ids, labels).
    """
    try:
        parsed = json.loads(cell)
    except Exception:
        return [], []

    ids: List[str] = []
    labels: List[str] = []
    if not isinstance(parsed, list):
        return ids, labels

    for item in parsed:
        if isinstance(item, list) and len(item) >= 2:
            ids.append(str(item[0]).strip())
            labels.append(str(item[1]).strip())
    return ids, labels


def _make_prompt(entry_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(entry=entry_text)},
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multilabel_dir",
        type=str,
        default="/data/muhsen/repos/verl/Datasets/train_multilabel",
        help="Directory containing multilabel *_train*.csv files.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="*_train_rl.csv",
        help="Glob pattern for multilabel CSV files to combine.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/workspace/verl/rl-data/",
        help="Directory to save parquet file",
    )
    parser.add_argument(
        "--out_parquet_name",
        default="multilabel_combined_train_rl.parquet",
        help="Output parquet filename (written into local_save_dir).",
    )
    args = parser.parse_args()

    multilabel_dir = Path(args.multilabel_dir)
    save_dir = Path(os.path.expanduser(args.local_save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    if not multilabel_dir.exists():
        raise FileNotFoundError(f"Multilabel dir not found: {multilabel_dir}")

    csv_paths = sorted(multilabel_dir.glob(args.glob_pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No files matched glob_pattern={args.glob_pattern} in {multilabel_dir}")

    rows: List[Dict[str, Any]] = []
    global_idx = 0

    # Load and combine CSVs.
    for csv_path in csv_paths:
        prefix = _prefix_from_filename(csv_path)
        dataset = datasets.load_dataset("csv", data_files=str(csv_path))["train"]

        required_cols = ["id", "sct_id", "entry"]
        for col in required_cols:
            if col not in dataset.column_names:
                raise ValueError(f"Missing required column {col!r} in {csv_path}. Found {dataset.column_names}")

        for ex in dataset:
            entry_text = str(ex["entry"]).strip()
            raw_id = str(ex["id"]).strip()
            prefixed_id = f"{prefix}{raw_id}"

            gt_sct_ids, gt_labels = _parse_multilabel_sct_id(str(ex["sct_id"]))

            rows.append(
                {
                    # Important: dataset identifier must NOT contain "snomed"
                    # so reward routing can use the multilabel reward function.
                    "data_source": "sct_multilabel",
                    "prompt": _make_prompt(entry_text),
                    "ability": "medical_coding",
                    "reward_model": {
                        "style": "rule",
                        # Keep `ground_truth` as a JSON string; reward manager will receive it as-is.
                        "ground_truth": json.dumps(
                            {"id": prefixed_id, "gt_sct_ids": gt_sct_ids, "gt_labels": gt_labels},
                            ensure_ascii=False,
                        ),
                    },
                    "extra_info": {
                        "index": global_idx,
                        "entry": entry_text,
                        "raw_id": raw_id,
                        "prefixed_id": prefixed_id,
                        "source_csv": csv_path.name,
                        "gt_sct_ids": gt_sct_ids,
                        "gt_labels": gt_labels,
                    },
                }
            )
            global_idx += 1

    combined = datasets.Dataset.from_list(rows)
    output_path = save_dir / args.out_parquet_name
    combined.to_parquet(str(output_path))

    print(f"Saved parquet to: {output_path}")
    print(f"Total examples: {len(combined)}")
