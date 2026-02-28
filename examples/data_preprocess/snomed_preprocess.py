# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Preprocess a SNOMED CSV dataset to parquet format
"""

import argparse
import os
import json
import datasets


def make_system_prompt() -> str:
    return """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

You are a medical coding assistant.

Provide the SNOMED CT concept ID (SCTID) and the corresponding SNOMED description

Now, analyze the following patient entry and provide your reasoning and answer.""".strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="CSV filename inside ./Datasets/ directory",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/snomed",
        help="Directory to save parquet file",
    )

    args = parser.parse_args()

    dataset_path = os.path.join("./Datasets", args.dataset_name)
    save_dir = os.path.expanduser(args.local_save_dir)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    os.makedirs(save_dir, exist_ok=True)

    # Load CSV using HF datasets
    dataset = datasets.load_dataset("csv", data_files=dataset_path)["train"]

    required_cols = ["entry", "sct_id", "label"]
    for col in required_cols:
        if col not in dataset.column_names:
            raise ValueError(f"Missing required column: {col}")

    system_prompt = make_system_prompt()

    def process_fn(example, idx):
        entry_text = str(example["entry"]).strip()
        sct_id = str(example["sct_id"]).strip()
        label = str(example["label"]).strip()

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry_text},
        ]

        ground_truth = {
            "id": f"{idx:012d}",
            "sct_id": sct_id,
            "label": label,
        }

        return {
            "data_source": args.dataset_name,
            "prompt": prompt_messages,
            "ability": "medical_coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            },
            "extra_info": {
                "index": idx,
                "entry": entry_text,
                "sct_id": sct_id,
                "label": label,
            },
        }

    dataset = dataset.map(function=process_fn, with_indices=True)

    # Save ONE parquet with same name as CSV
    parquet_name = args.dataset_name.replace(".csv", ".parquet")
    output_path = os.path.join(save_dir, parquet_name)

    dataset.to_parquet(output_path)

    print(f"Saved parquet to: {output_path}")
    print(f"Total examples: {len(dataset)}")
