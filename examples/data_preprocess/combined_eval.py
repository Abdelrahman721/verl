"""Preprocess and combine evaluation data from multiple domains into a single parquet.

Evaluation sources:
  Math:
    1. GSM8K test split           – openai/gsm8k
    2. MATH-500 test split        – HuggingFaceH4/MATH-500
  IF:
    3. IFEval test split          – google/IFEval
    4. IFBench_test train split   – allenai/IFBench_test

General chat is excluded (manual evaluation).

Output: a single eval.parquet in the verl-compatible format.
"""

import argparse
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
MATH_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def extract_gsm8k_solution(answer_str):
    """Extract the numeric answer after #### from GSM8K."""
    match = re.search(r"####\s*([\-\d.,]+)", answer_str)
    assert match is not None, f"Could not extract solution from: {answer_str[:100]}"
    return match.group(1).replace(",", "").strip()


def extract_math_solution(solution_str):
    """Extract the answer from a \\boxed{} expression."""
    return remove_boxed(last_boxed_only_string(solution_str))


def process_gsm8k(dataset_name="openai/gsm8k", local_path=None):
    """Load and process GSM8K test split."""
    print(f"[Eval-GSM8K] Loading {dataset_name} ...", flush=True)
    if local_path:
        ds = datasets.load_dataset(local_path, "main", split="test")
    else:
        ds = datasets.load_dataset(dataset_name, "main", split="test")

    def map_fn(example, idx):
        question = example["question"].strip() + " " + GSM8K_INSTRUCTION
        solution = extract_gsm8k_solution(example["answer"])

        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "test",
                "index": idx,
                "answer": example["answer"],
                "question": example["question"],
            },
        }

    return ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)


def process_math500(dataset_name="HuggingFaceH4/MATH-500", local_path=None):
    """Load and process MATH-500 test split."""
    print(f"[Eval-MATH500] Loading {dataset_name} ...", flush=True)
    if local_path:
        ds = datasets.load_dataset(local_path, split="test")
    else:
        ds = datasets.load_dataset(dataset_name, split="test")

    problem_col = "problem" if "problem" in ds.column_names else "question"
    has_clean_answer = "answer" in ds.column_names
    if has_clean_answer:
        answer_col = "answer"
    elif "solution" in ds.column_names:
        answer_col = "solution"
    else:
        raise KeyError(f"Cannot find answer column in {ds.column_names}")

    def map_fn(example, idx):
        question = example[problem_col].strip() + " " + MATH_INSTRUCTION
        raw_answer = example[answer_col]
        if has_clean_answer:
            solution = raw_answer.strip()
        else:
            try:
                solution = extract_math_solution(raw_answer)
            except Exception:
                solution = raw_answer.strip()

        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "test",
                "index": idx,
                "subject": example.get("subject", ""),
                "level": example.get("level", ""),
            },
        }

    return ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)


def _build_if_ground_truth(instruction_id_list, kwargs_list):
    """Convert IFEval-style columns into the ground_truth string the reward fn expects.

    Returns a string representation of:
      [{"instruction_id": [...], "kwargs": [...]}]
    """
    if kwargs_list is None:
        kwargs_list = [None] * len(instruction_id_list)
    constraint_group = {
        "instruction_id": list(instruction_id_list),
        "kwargs": list(kwargs_list),
    }
    return str([constraint_group])


def process_ifeval(dataset_name="google/IFEval", local_path=None):
    """Load and process IFEval test (train split on HF)."""
    print(f"[Eval-IFEval] Loading {dataset_name} ...", flush=True)
    if local_path:
        ds = datasets.load_dataset(local_path, split="train")
    else:
        ds = datasets.load_dataset(dataset_name, split="train")

    def map_fn(example, idx):
        gt = _build_if_ground_truth(
            example["instruction_id_list"],
            example["kwargs"],
        )

        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": example["prompt"].strip()}],
            "ability": "instruction_following",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "test",
                "index": idx,
                "key": example.get("key", ""),
            },
        }

    return ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)


def process_ifbench(dataset_name="allenai/IFBench_test", local_path=None):
    """Load and process IFBench_test train split."""
    print(f"[Eval-IFBench] Loading {dataset_name} ...", flush=True)
    if local_path:
        ds = datasets.load_dataset(local_path, split="train")
    else:
        ds = datasets.load_dataset(dataset_name, split="train")

    def map_fn(example, idx):
        gt = _build_if_ground_truth(
            example["instruction_id_list"],
            example["kwargs"],
        )

        return {
            "data_source": dataset_name,
            "prompt": [{"role": "user", "content": example["prompt"].strip()}],
            "ability": "instruction_following",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "test",
                "index": idx,
                "key": example.get("key", ""),
            },
        }

    return ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multi-domain eval data.")
    parser.add_argument(
        "--local_save_dir",
        default="/data/abdelrahman/verl/data/combined",
        help="Directory to write the combined eval parquet.",
    )
    parser.add_argument("--gsm8k_local_path", default=None)
    parser.add_argument("--math500_local_path", default=None)
    parser.add_argument("--ifeval_local_path", default=None)
    parser.add_argument("--ifbench_local_path", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    gsm8k_ds = process_gsm8k(local_path=args.gsm8k_local_path)
    math500_ds = process_math500(local_path=args.math500_local_path)
    ifeval_ds = process_ifeval(local_path=args.ifeval_local_path)
    ifbench_ds = process_ifbench(local_path=args.ifbench_local_path)

    print(
        f"\n[Summary] GSM8K: {len(gsm8k_ds)}, MATH-500: {len(math500_ds)}, "
        f"IFEval: {len(ifeval_ds)}, IFBench: {len(ifbench_ds)}"
    )

    combined = datasets.concatenate_datasets([gsm8k_ds, math500_ds, ifeval_ds, ifbench_ds])
    print(f"[Summary] Combined eval set: {len(combined)} examples")

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    out_path = os.path.join(local_dir, "eval.parquet")
    combined.to_parquet(out_path)
    print(f"[Done] Saved to {out_path}")

    example = combined[0]
    with open(os.path.join(local_dir, "eval_example.json"), "w") as f:
        json.dump(example, f, indent=2, default=str)

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
