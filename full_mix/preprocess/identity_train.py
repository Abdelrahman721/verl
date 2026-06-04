"""Preprocess the Avey identity dataset (local disk) for VERL.

Each source row is an identity-related prompt (often adversarial — user
tries to relabel the assistant as ChatGPT/HealthGPT/etc., claims admin
override, etc.) plus a list of ``reference_responses`` that demonstrate
the correct identity-preserving reply.

We pick the FIRST reference as the canonical ground_truth (the identity
reward judge uses it as the reference example). The remaining context
fields (mode/category/topic) are kept under extra_info for downstream
analysis but are not used by the reward at training time.
"""

import argparse
import json
import os

import datasets

from full_mix.preprocess._common import ensure_dir


DATA_SOURCE = "local/avey-identity"
DEFAULT_SRC = "/data/abdelrahman/qwen-sft/rl-data-prep/data/identity"


def _pick_reference(refs) -> str:
    if isinstance(refs, list):
        if not refs:
            return ""
        return str(refs[0]).strip()
    if refs is None:
        return ""
    return str(refs).strip()


def build(src: str, max_rows: int | None = None):
    ds = datasets.load_from_disk(src)
    if max_rows is not None:
        ds = ds.shuffle(seed=42).select(range(min(max_rows, len(ds))))

    def map_fn(example, idx):
        content = (example.get("prompt") or "").strip()
        ref = _pick_reference(example.get("reference_responses"))
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": content}],
            "ability": "identity",
            "reward_model": {"style": "rule", "ground_truth": ref},
            "extra_info": {
                "split": "train",
                "index": idx,
                "user_prompt": content,
                "mode": str(example.get("mode") or ""),
                "category": str(example.get("category") or ""),
                "topic": str(example.get("topic") or ""),
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
    out_path = os.path.join(out_dir, "identity_train.parquet")

    ds = build(args.src, max_rows=args.max_rows)
    ds.to_parquet(out_path)
    print(f"[identity_train] wrote {len(ds)} rows to {out_path}")

    with open(os.path.join(out_dir, "identity_train_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, default=str)


if __name__ == "__main__":
    main()
