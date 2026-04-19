"""Preprocess 100 samples of GSM8K test split for VERL evaluation."""

import argparse
import json
import os
import re

import datasets

from full_mix.preprocess._common import GSM8K_INSTRUCTION, ensure_dir


DATA_SOURCE = "openai/gsm8k"
N_EVAL = 100


def _extract(answer_str: str) -> str:
    m = re.search(r"####\s*(\-?[0-9\.,]+)", answer_str)
    assert m is not None, f"no #### in: {answer_str[:120]!r}"
    return m.group(1).replace(",", "").strip()


def build(local_path: str | None, n: int = N_EVAL):
    if local_path:
        ds = datasets.load_dataset(local_path, "main", split="test")
    else:
        ds = datasets.load_dataset(DATA_SOURCE, "main", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    def map_fn(example, idx):
        q = example["question"].strip() + " " + GSM8K_INSTRUCTION
        gt = _extract(example["answer"])
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": q}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "test",
                "index": idx,
                "question": example["question"],
                "answer": example["answer"],
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
    out_path = os.path.join(out_dir, "gsm8k_eval.parquet")
    ds = build(args.local_path, n=args.n)
    ds.to_parquet(out_path)
    print(f"[gsm8k_eval] wrote {len(ds)} rows to {out_path}")

    with open(os.path.join(out_dir, "gsm8k_eval_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, default=str)


if __name__ == "__main__":
    main()
