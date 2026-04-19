"""Preprocess 100 samples of MATH-500 test split for VERL evaluation."""

import argparse
import json
import os

import datasets

from full_mix.preprocess._common import MATH_INSTRUCTION, ensure_dir


DATA_SOURCE = "HuggingFaceH4/MATH-500"
N_EVAL = 100


def build(local_path: str | None, n: int = N_EVAL):
    if local_path:
        ds = datasets.load_dataset(local_path, split="test")
    else:
        ds = datasets.load_dataset(DATA_SOURCE, split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    problem_col = "problem" if "problem" in ds.column_names else "question"
    answer_col = "answer" if "answer" in ds.column_names else "solution"

    def map_fn(example, idx):
        q = example[problem_col].strip() + " " + MATH_INSTRUCTION
        gt = str(example[answer_col]).strip()
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": q}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "test",
                "index": idx,
                "subject": example.get("subject", "") or "",
                "level": example.get("level", "") or "",
            },
        }

    return ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_path", default=None)
    ap.add_argument("--out_dir", default="/data/abdelrahman/verl/data/full_mix/eval")
    ap.add_argument("--n", type=int, default=N_EVAL)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    out_path = os.path.join(out_dir, "math500_eval.parquet")
    ds = build(args.local_path, n=args.n)
    ds.to_parquet(out_path)
    print(f"[math500_eval] wrote {len(ds)} rows to {out_path}")

    with open(os.path.join(out_dir, "math500_eval_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, default=str)


if __name__ == "__main__":
    main()
