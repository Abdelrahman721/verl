"""Preprocess and combine training data from three domains into a single parquet.

Domains:
  1. Math       – allenai/Dolci-RL-Zero-Math-7B
  2. IF         – allenai/Dolci-RL-Zero-IF-7B
  3. General    – allenai/Dolci-RL-Zero-General-7B

Output: a single train.parquet in the verl-compatible format.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

MATH_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def process_math(dataset_name="allenai/Dolci-RL-Zero-Math-7B", local_path=None):
    """Load and process the Dolci Math training data."""
    print(f"[Math] Loading {dataset_name} ...", flush=True)
    if local_path:
        ds = datasets.load_dataset(local_path, split="train")
    else:
        ds = datasets.load_dataset(dataset_name, split="train")

    def map_fn(example, idx):
        messages = example.get("messages")
        if messages and len(messages) > 0:
            content = messages[0]["content"]
        else:
            content = example["prompt"]

        content = content.strip() + " " + MATH_INSTRUCTION

        gt = example["ground_truth"]

        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(gt)},
            "extra_info": {
                "split": "train",
                "index": idx,
                "original_dataset": example.get("dataset", "math"),
            },
        }

    return ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)


def process_if(dataset_name="allenai/Dolci-RL-Zero-IF-7B", local_path=None):
    """Load and process the Dolci IF training data."""
    print(f"[IF] Loading {dataset_name} ...", flush=True)
    if local_path:
        ds = datasets.load_dataset(local_path, split="train")
    else:
        ds = datasets.load_dataset(dataset_name, split="train")

    def map_fn(example, idx):
        raw_prompt = example["prompt"]
        if raw_prompt.startswith("user: "):
            content = raw_prompt[len("user: "):]
        elif raw_prompt.startswith("user:"):
            content = raw_prompt[len("user:"):]
        else:
            content = raw_prompt

        gt_list = example["ground_truth"]
        if isinstance(gt_list, list) and len(gt_list) > 0:
            gt = gt_list[0]
        else:
            gt = str(gt_list)

        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": content.strip()}],
            "ability": "instruction_following",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "train",
                "index": idx,
                "key": example.get("key", ""),
                "constraint": example.get("constraint", ""),
            },
        }

    return ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)


def process_general(dataset_name="allenai/Dolci-RL-Zero-General-7B", local_path=None):
    """Load and process the Dolci General training data."""
    print(f"[General] Loading {dataset_name} ...", flush=True)
    if local_path:
        ds = datasets.load_dataset(local_path, split="train")
    else:
        ds = datasets.load_dataset(dataset_name, split="train")

    def map_fn(example, idx):
        raw_prompt = example["prompt"]
        if raw_prompt.startswith("user: "):
            content = raw_prompt[len("user: "):]
        elif raw_prompt.startswith("user:"):
            content = raw_prompt[len("user:"):]
        else:
            content = raw_prompt

        gt_list = example["ground_truth"]
        if isinstance(gt_list, list) and len(gt_list) > 0:
            gt = gt_list[0]
        else:
            gt = str(gt_list)

        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": content.strip()}],
            "ability": "general_chat",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "train",
                "index": idx,
                "custom_id": example.get("custom_id", ""),
            },
        }

    return ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multi-domain training data for GRPO.")
    parser.add_argument(
        "--local_save_dir",
        default="/data/abdelrahman/verl/data/combined",
        help="Directory to write the combined parquet files.",
    )
    parser.add_argument("--math_local_path", default=None, help="Local path to pre-downloaded Math dataset.")
    parser.add_argument("--if_local_path", default=None, help="Local path to pre-downloaded IF dataset.")
    parser.add_argument("--general_local_path", default=None, help="Local path to pre-downloaded General dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    math_ds = process_math(local_path=args.math_local_path)
    if_ds = process_if(local_path=args.if_local_path)
    general_ds = process_general(local_path=args.general_local_path)

    print(f"\n[Summary] Math: {len(math_ds)}, IF: {len(if_ds)}, General: {len(general_ds)}")

    combined = datasets.concatenate_datasets([math_ds, if_ds, general_ds])
    combined = combined.shuffle(seed=42)
    print(f"[Summary] Combined training set: {len(combined)} examples")

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    out_path = os.path.join(local_dir, "train.parquet")
    combined.to_parquet(out_path)
    print(f"[Done] Saved to {out_path}")

    example = combined[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2, default=str)

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
