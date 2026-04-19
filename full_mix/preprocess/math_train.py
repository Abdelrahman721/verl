"""Preprocess Dolci-RL-Zero-Math-7B (local disk format) for VERL."""

import argparse
import json
import os

import datasets

from full_mix.preprocess._common import MATH_INSTRUCTION, ensure_dir


DATA_SOURCE = "local/dolci-math-7b"
DEFAULT_SRC = "/data/abdelrahman/qwen-sft/rl-data-prep/data/dolci_rl_zero_math_7b"


def build(src: str, max_rows: int | None = None):
    ds = datasets.load_from_disk(src)
    if max_rows is not None:
        ds = ds.shuffle(seed=42).select(range(min(max_rows, len(ds))))

    def map_fn(example, idx):
        prompt = example.get("prompt") or ""
        content = prompt.strip() + " " + MATH_INSTRUCTION
        gt = example.get("ground_truth")
        return {
            "data_source": DATA_SOURCE,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out_dir", default="/data/abdelrahman/verl/data/full_mix/train")
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    out_path = os.path.join(out_dir, "math_train.parquet")

    ds = build(args.src, max_rows=args.max_rows)
    ds.to_parquet(out_path)
    print(f"[math_train] wrote {len(ds)} rows to {out_path}")

    with open(os.path.join(out_dir, "math_train_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, default=str)


if __name__ == "__main__":
    main()
