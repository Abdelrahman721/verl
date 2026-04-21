# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Preprocess instruction-following SNOMED JSONL (combined_dataset.jsonl from mcs-instruct-dataset)
to parquet format for verl.

Expects each line to be JSON with:
  - prompt: chat messages; the training prompt is taken only from messages with role \"user\".
  - metadata: rule, original_codes, processed_codes (and optionally original_text, etc.).

data_source is fixed to \"sct_if\". ground_truth is JSON with rule, original_codes, processed_codes (no id).
extra_info includes user_prompt (concatenated user message text) for reward rules that need the prompt (e.g. v10).
"""

from __future__ import annotations

import argparse
import json
import os

import datasets


DATA_SOURCE = "sct_if"


def user_messages_only(prompt: list[dict]) -> list[dict[str, str]]:
    """Keep only user-role messages (content strings) for the parquet prompt field."""
    out: list[dict[str, str]] = []
    for m in prompt:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if content is None:
            raise ValueError('user message missing "content"')
        out.append({"role": "user", "content": str(content)})
    if not out:
        raise ValueError("prompt has no messages with role=user")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        required=True,
        help='JSONL filename inside ./Datasets/ (e.g. train_instructionfollowing/combined_dataset.jsonl)',
    )
    parser.add_argument(
        "--local_save_dir",
        default="/workspace/verl/rl-data/",
        help="Directory to save parquet file",
    )

    args = parser.parse_args()

    dataset_path = os.path.join("./Datasets", args.dataset_name)
    save_dir = os.path.expanduser(args.local_save_dir)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    os.makedirs(save_dir, exist_ok=True)

    dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]

    def process_fn(example, idx):
        raw_prompt = example.get("prompt")
        if raw_prompt is None:
            raise ValueError("missing prompt")
        prompt_messages = user_messages_only(list(raw_prompt))

        meta = example.get("metadata")
        if meta is None:
            raise ValueError("missing metadata")
        if isinstance(meta, str):
            meta = json.loads(meta)

        rule = meta.get("rule")
        original_codes = meta.get("original_codes")
        processed_codes = meta.get("processed_codes")
        if rule is None or original_codes is None or processed_codes is None:
            raise ValueError(
                f"metadata must include rule, original_codes, processed_codes (example index {idx})"
            )

        ground_truth = {
            "rule": rule,
            "original_codes": original_codes,
            "processed_codes": processed_codes,
        }

        user_prompt = "\n\n".join(str(m["content"]) for m in prompt_messages)
        extra = {"index": idx, "rule": rule, "user_prompt": user_prompt}
        if meta.get("original_text") is not None:
            extra["original_text"] = meta["original_text"]

        return {
            "data_source": DATA_SOURCE,
            "prompt": prompt_messages,
            "ability": "instruction_following",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            },
            "extra_info": extra,
        }

    dataset = dataset.map(function=process_fn, with_indices=True)

    base = os.path.basename(args.dataset_name)
    parquet_name = base.replace(".jsonl", ".parquet")
    if parquet_name == base:
        parquet_name = base + ".parquet"
    output_path = os.path.join(save_dir, parquet_name)

    dataset.to_parquet(output_path)

    print(f"Saved parquet to: {output_path}")
    print(f"data_source: {DATA_SOURCE}")
    print(f"Total examples: {len(dataset)}")
