"""Preprocess dolci_think_rl_32b_chat (local disk) for VERL.

Ground truth is a short factual reference; the chat LLM judge uses it only
as factual grounding. The user prompt is tucked under ``extra_info`` so the
judge can see the original question when scoring.
"""

import argparse
import json
import os

import datasets

from full_mix.preprocess._common import ensure_dir, strip_user_prefix


DATA_SOURCE = "local/dolci-chat-32b"
DEFAULT_SRC = "/data/abdelrahman/qwen-sft/rl-data-prep/data/dolci_think_rl_32b_chat"


def _pick_reference(raw) -> str:
    if isinstance(raw, list):
        if not raw:
            return ""
        return str(raw[0])
    return str(raw) if raw is not None else ""


def build(src: str, max_rows: int | None = None):
    ds = datasets.load_from_disk(src)
    if max_rows is not None:
        ds = ds.shuffle(seed=42).select(range(min(max_rows, len(ds))))

    def map_fn(example, idx):
        content = strip_user_prefix(example.get("prompt") or "").strip()
        ref = _pick_reference(example.get("ground_truth"))
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": content}],
            "ability": "chat",
            "reward_model": {"style": "rule", "ground_truth": ref},
            "extra_info": {
                "split": "train",
                "index": idx,
                "user_prompt": content,
                "custom_id": example.get("custom_id", "") or "",
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
    out_path = os.path.join(out_dir, "chat_train.parquet")

    ds = build(args.src, max_rows=args.max_rows)
    ds.to_parquet(out_path)
    print(f"[chat_train] wrote {len(ds)} rows to {out_path}")

    with open(os.path.join(out_dir, "chat_train_example.json"), "w") as f:
        json.dump(ds[0], f, indent=2, default=str)


if __name__ == "__main__":
    main()
