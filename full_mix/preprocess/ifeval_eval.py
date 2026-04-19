"""Preprocess 100 samples of google/IFEval for VERL evaluation.

Wraps ``instruction_id_list`` + ``kwargs`` into the ``[{instruction_id, kwargs}]``
constraint-list format that the ifeval reward function expects.
"""

import argparse
import json
import os

import datasets

from full_mix.preprocess._common import ensure_dir


DATA_SOURCE = "google/IFEval"
N_EVAL = 100


def _build_gt(instruction_id_list, kwargs_list) -> str:
    if kwargs_list is None:
        kwargs_list = [None] * len(instruction_id_list)
    return str([{"instruction_id": list(instruction_id_list), "kwargs": list(kwargs_list)}])


def build(local_path: str | None, n: int = N_EVAL):
    if local_path:
        ds = datasets.load_dataset(local_path, split="train")
    else:
        ds = datasets.load_dataset(DATA_SOURCE, split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    def map_fn(example, idx):
        gt = _build_gt(example["instruction_id_list"], example.get("kwargs"))
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": example["prompt"].strip()}],
            "ability": "instruction_following",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "test",
                "index": idx,
                "key": str(example.get("key", "")) if example.get("key") is not None else "",
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
    out_path = os.path.join(out_dir, "ifeval_eval.parquet")
    ds = build(args.local_path, n=args.n)
    ds.to_parquet(out_path)
    print(f"[ifeval_eval] wrote {len(ds)} rows to {out_path}")

    with open(os.path.join(out_dir, "ifeval_eval_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, default=str)


if __name__ == "__main__":
    main()
