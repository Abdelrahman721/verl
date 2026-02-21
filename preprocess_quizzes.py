"""
Preprocess the medical quiz dataset (clean-quizzes-new/) to parquet format
compatible with verl's GRPO training pipeline.

Each JSON file contains a title and a list of quiz items with:
- question: the clinical question
- answer: the gold reference answer
- rubric: list of grading criteria strings

The output parquet has the verl schema:
- data_source, prompt, ability, reward_model, extra_info
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def load_quizzes(quiz_dir: str) -> list[dict]:
    """Load all quiz JSON files and flatten into individual Q&A items."""
    items = []
    quiz_dir = Path(quiz_dir)
    for json_file in sorted(quiz_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        title = data.get("title", "")
        for i, q in enumerate(data.get("quiz", [])):
            items.append({
                "source_file": json_file.name,
                "title": title,
                "question": q["question"],
                "answer": q["answer"],
                "rubric": q["rubric"],
                "quiz_index": i,
            })
    return items


SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following clinical question thoroughly and accurately.\n"
    "Think step by step inside <think>...</think> tags, then provide your final answer directly after the closing </think> tag."
)


def build_dataset(items: list[dict], split: str = "train") -> list[dict]:
    """Convert raw quiz items into verl-compatible dataset rows."""
    rows = []
    for idx, item in enumerate(items):
        row = {
            "data_source": "medical_qa",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ],
            "ability": "medical_qa",
            "reward_model": {
                "style": "llm_judge",
                "ground_truth": {
                    "gold_answer": item["answer"],
                    "criteria": item["rubric"],
                },
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "title": item["title"],
                "source_file": item["source_file"],
                "quiz_index": item["quiz_index"],
            },
        }
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess medical quiz dataset to parquet")
    parser.add_argument(
        "--quiz_dir",
        default="clean-quizzes-new",
        help="Directory containing quiz JSON files",
    )
    parser.add_argument(
        "--output_dir",
        default="data/medical_qa",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    print(f"Loading quizzes from {args.quiz_dir}...")
    items = load_quizzes(args.quiz_dir)
    print(f"Found {len(items)} quiz items from {len(list(Path(args.quiz_dir).glob('*.json')))} files")

    # Shuffle and split
    import random
    random.seed(args.seed)
    random.shuffle(items)

    val_count = max(1, int(len(items) * args.val_ratio))
    val_items = items[:val_count]
    train_items = items[val_count:]

    print(f"Train: {len(train_items)} items, Val: {val_count} items")

    train_rows = build_dataset(train_items, split="train")
    val_rows = build_dataset(val_items, split="val")

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"Saved train to {train_path}")
    print(f"Saved val to {val_path}")

    # Print a sample
    sample = train_rows[0]
    print("\n--- Sample row ---")
    print(f"data_source: {sample['data_source']}")
    print(f"prompt: {json.dumps(sample['prompt'], indent=2)[:300]}...")
    print(f"ground_truth keys: {list(sample['reward_model']['ground_truth'].keys())}")
    print(f"criteria count: {len(sample['reward_model']['ground_truth']['criteria'])}")
