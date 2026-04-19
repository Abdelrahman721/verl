"""Preprocess dolci_think_rl_32b_ifeval (local disk) for VERL.

The local dataset's ``ground_truth`` is ``List[str]`` where element 0 is a
Python-literal representation of the IFEval constraint list:
``"[{'instruction_id': [...], 'kwargs': [...]}]"``. We pass that string
through unchanged — the reward function does ``ast.literal_eval`` later.
"""

import argparse
import ast
import json
import os

import datasets

from full_mix.preprocess._common import ensure_dir, strip_user_prefix


DATA_SOURCE = "local/dolci-ifeval-32b"
DEFAULT_SRC = "/data/abdelrahman/qwen-sft/rl-data-prep/data/dolci_think_rl_32b_ifeval"


def _pick_ground_truth(raw) -> str:
    if isinstance(raw, list):
        if not raw:
            return ""
        gt = raw[0]
    else:
        gt = raw
    # Soft validation: if it doesn't parse as a constraint list, keep the
    # string anyway — the reward fn will log and skip at score-time.
    if isinstance(gt, str):
        try:
            ast.literal_eval(gt)
        except (ValueError, SyntaxError):
            pass
    return str(gt) if gt is not None else ""


def build(src: str, max_rows: int | None = None):
    ds = datasets.load_from_disk(src)
    if max_rows is not None:
        ds = ds.shuffle(seed=42).select(range(min(max_rows, len(ds))))

    def map_fn(example, idx):
        content = strip_user_prefix(example.get("prompt") or "").strip()
        gt_str = _pick_ground_truth(example.get("ground_truth"))
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": content}],
            "ability": "instruction_following",
            "reward_model": {"style": "rule", "ground_truth": gt_str},
            "extra_info": {
                "split": "train",
                "index": idx,
                "key": example.get("key", "") or "",
                "constraint": example.get("constraint", "") or "",
                "constraint_type": example.get("constraint_type", "") or "",
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
    out_path = os.path.join(out_dir, "ifeval_train.parquet")

    ds = build(args.src, max_rows=args.max_rows)
    ds.to_parquet(out_path)
    print(f"[ifeval_train] wrote {len(ds)} rows to {out_path}")

    with open(os.path.join(out_dir, "ifeval_train_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, default=str)


if __name__ == "__main__":
    main()
