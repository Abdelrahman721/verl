"""Preprocess safety_dpo_with_reference (local disk) for VERL."""

import argparse
import json
import os

import datasets

from full_mix.preprocess._common import ensure_dir


DATA_SOURCE = "local/safety-dpo-reference"
DEFAULT_SRC = "/data/abdelrahman/qwen-sft/rl-data-prep/data/safety_dpo_with_reference"


def build(src: str, max_rows: int | None = None):
    ds = datasets.load_from_disk(src)
    if max_rows is not None:
        ds = ds.shuffle(seed=42).select(range(min(max_rows, len(ds))))

    def map_fn(example, idx):
        content = (example.get("prompt") or "").strip()
        ref = (example.get("reference_answer") or "").strip()
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": content}],
            "ability": "safety",
            "reward_model": {"style": "rule", "ground_truth": ref},
            "extra_info": {
                "split": "train",
                "index": idx,
                "user_prompt": content,
                "tag": example.get("tag", "") or "",
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
    out_path = os.path.join(out_dir, "safety_train.parquet")

    ds = build(args.src, max_rows=args.max_rows)
    ds.to_parquet(out_path)
    print(f"[safety_train] wrote {len(ds)} rows to {out_path}")

    with open(os.path.join(out_dir, "safety_train_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, default=str)


if __name__ == "__main__":
    main()
