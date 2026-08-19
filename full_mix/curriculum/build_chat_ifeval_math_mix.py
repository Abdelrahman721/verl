"""Build a subsampled chat + ifeval + math training mix.

Draws a fixed number of prompts uniformly at random (without replacement) from
each of the three full_mix training parquets and writes one subsampled parquet
per source:

    chat   <- chat_with_baseline_train.parquet   (default 2000 prompts)
    ifeval <- ifeval_train.parquet               (default 4000 prompts)
    math   <- math_train.parquet                 (default 2000 prompts)

Each output preserves the EXACT schema of its source: rows are sliced via
``pq.read_table().take([indices])`` so struct types, column order and schema
metadata are bit-identical. The three files can be listed directly in
``data.train_files`` — verl concatenates them with HF ``datasets`` and shuffles
(``data.shuffle=True``), so no merged parquet is needed.

Row identity is preserved too. ``extra_info.index`` equals the row's position
in the *source* parquet (set at preprocessing time with ``with_indices=True``),
and the reward dispatcher derives

    prompt_uid = DATASET_ID[base_data_source] * UID_DATASET_STRIDE + index

from it. Because we only ``.take()`` rows — never renumber — a prompt keeps the
same uid it had in the full dataset, so rollout dumps from this run stay
compatible with ``build_curated_subset.py`` and the stats scripts.

Sampling is deterministic and *independent per source*: each source gets its own
RNG seeded with ``seed + DATASET_ID``, so changing one source's count leaves the
other two subsets unchanged.

Usage:
    python -m full_mix.curriculum.build_chat_ifeval_math_mix
    python -m full_mix.curriculum.build_chat_ifeval_math_mix \\
        --n_chat 2000 --n_ifeval 4000 --n_math 2000 \\
        --output_dir /data/abdelrahman/verl/data/chat_ifeval_math_mix \\
        --seed 42
"""

import argparse
import json
import os
import random

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from full_mix.curriculum.encoding import DATASET_ID, UID_DATASET_STRIDE

# base data_source -> (source parquet filename, CLI arg holding the count)
SOURCES: dict[str, tuple[str, str]] = {
    "local/dolci-chat-32b": ("chat_with_baseline_train.parquet", "n_chat"),
    "local/dolci-ifeval-32b": ("ifeval_train.parquet", "n_ifeval"),
    "local/dolci-math-7b": ("math_train.parquet", "n_math"),
}


def sample_indices(n_rows: int, n_wanted: int, seed: int) -> list[int]:
    """Uniform sample of `n_wanted` distinct row indices from [0, n_rows).

    Returns them sorted so the output parquet keeps the source's row order
    (the training dataloader shuffles anyway).
    """
    if n_wanted > n_rows:
        raise SystemExit(f"requested {n_wanted} prompts but source only has {n_rows} rows")
    if n_wanted == n_rows:
        return list(range(n_rows))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_rows), n_wanted))


def check_row_identity(table: pa.Table, indices: list[int], base_ds: str) -> None:
    """Assert every sampled row still carries its ORIGINAL source row index.

    ``extra_info.index`` is what the reward dispatcher turns into ``prompt_uid``.
    If a source parquet were ever rebuilt without ``with_indices=True``, the
    uids in this run's rollout dumps would silently point at the wrong rows —
    so fail loudly here instead.
    """
    got = pc.struct_field(table.column("extra_info"), "index").to_pylist()
    if got != indices:
        bad = next((i for i, (a, b) in enumerate(zip(got, indices)) if a != b), 0)
        raise SystemExit(
            f"{base_ds}: extra_info.index does not match source row position "
            f"(row {bad}: extra_info.index={got[bad]}, source row={indices[bad]}). "
            f"Re-run the preprocessor with with_indices=True."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_train_dir", default="/data/abdelrahman/verl/data/full_mix/train")
    ap.add_argument("--output_dir", default="/data/abdelrahman/verl/data/chat_ifeval_math_mix")
    ap.add_argument("--n_chat", type=int, default=2000)
    ap.add_argument("--n_ifeval", type=int, default=4000)
    ap.add_argument("--n_math", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--chat_variant",
        choices=["chat", "chat_with_baseline"],
        default="chat_with_baseline",
        help="Which chat parquet to sample from. The pairwise chat reward "
        "(full_mix/rewards/chat.py) needs extra_info.baseline_response, which "
        "only the chat_with_baseline variant carries.",
    )
    args = ap.parse_args()

    sources = dict(SOURCES)
    if args.chat_variant == "chat":
        sources["local/dolci-chat-32b"] = ("chat_train.parquet", "n_chat")

    os.makedirs(args.output_dir, exist_ok=True)

    summary: dict[str, dict] = {}
    for base_ds, (filename, count_arg) in sources.items():
        n_wanted = getattr(args, count_arg)
        dataset_id = DATASET_ID[base_ds]
        src = os.path.join(args.source_train_dir, filename)
        dst = os.path.join(args.output_dir, filename)

        table = pq.read_table(src)
        indices = sample_indices(table.num_rows, n_wanted, seed=args.seed + dataset_id)
        sub = table.take(pa.array(indices, type=pa.int64()))
        check_row_identity(sub, indices, base_ds)
        pq.write_table(sub, dst)

        uids = [dataset_id * UID_DATASET_STRIDE + i for i in indices]
        print(f"\n{base_ds}  (dataset_id={dataset_id})")
        print(f"  source        : {src}  ({table.num_rows} rows)")
        print(f"  sampled       : {sub.num_rows} rows  (seed={args.seed + dataset_id})")
        print(f"  prompt_uid    : {uids[0]} .. {uids[-1]}")
        print(f"  wrote ->      : {dst}")

        summary[base_ds] = {
            "filename": filename,
            "source": src,
            "output": dst,
            "source_rows": table.num_rows,
            "sampled_rows": sub.num_rows,
            "seed": args.seed + dataset_id,
            "row_indices": indices,
        }

    total = sum(s["sampled_rows"] for s in summary.values())
    print(f"\ntotal prompts in mix: {total}")

    report_path = os.path.join(args.output_dir, "_build_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "source_train_dir": args.source_train_dir,
                "chat_variant": args.chat_variant,
                "seed": args.seed,
                "total_prompts": total,
                "by_source": summary,
            },
            f,
            indent=2,
        )
    print(f"wrote report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
