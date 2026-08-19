"""Split each full_mix train parquet into NUM_SHARDS disjoint shards.

Reads the existing per-source parquet (ifeval / chat / safety), assigns each
row to a shard via a seeded permutation, suffixes the row's ``data_source``
with ``#shard{N}``, and stamps a globally-unique int ``uid`` plus
``origin_shard`` and empty ``carry_history`` into ``extra_info``.

Outputs:
    data/full_mix/train/shards/raw/shard_{N}/{name}_train.parquet   for N in 1..4
    data/full_mix/train/shards/stage_1/{name}_train.parquet         (= raw/shard_1/)

Stages 2..4 are produced later by full_mix/curriculum/build_next_stage.py
after the corresponding stage's training run finishes.
"""

import argparse
import json
import os
import shutil

import datasets
import numpy as np
import pyarrow.parquet as pq

from full_mix.curriculum.encoding import (
    NUM_SHARDS,
    make_uid,
    with_shard,
)
from full_mix.preprocess._common import ensure_dir


DEFAULT_TRAIN_DIR = "/data/abdelrahman/verl/data/full_mix/train"

# Both chat parquets share data_source="local/dolci-chat-32b"; they differ only
# in extra_info (chat_with_baseline_train.parquet adds baseline_model /
# baseline_response). Pick which one to shard via --chat-variant.
CHAT_VARIANTS = {
    "chat":               "chat_train.parquet",
    "chat_with_baseline": "chat_with_baseline_train.parquet",
}

# (filename, base_data_source). Filled in at runtime once the chat variant is
# chosen. We always emit the per-source parquet under its ORIGINAL filename
# inside each shard dir, so downstream code reads e.g.
# shards/stage_1/chat_with_baseline_train.parquet when that variant is chosen.
def _sources_for(chat_variant: str) -> list[tuple[str, str]]:
    if chat_variant not in CHAT_VARIANTS:
        raise SystemExit(
            f"--chat-variant must be one of {list(CHAT_VARIANTS)}, got {chat_variant!r}"
        )
    return [
        ("ifeval_train.parquet",         "local/dolci-ifeval-32b"),
        (CHAT_VARIANTS[chat_variant],    "local/dolci-chat-32b"),
        ("safety_train.parquet",         "local/safety-dpo-reference"),
        ("identity_train.parquet",       "local/avey-identity"),
    ]


# Re-exported for build_next_stage.py: defaults to the plain chat variant so
# imports succeed without args. build_next_stage takes its own --chat-variant.
SOURCES = _sources_for("chat")


def _shard_assignments(n_rows: int, num_shards: int, seed: int) -> np.ndarray:
    """Return shape-(n_rows,) int array of shard ids in [1, num_shards].

    Deterministic for a given ``seed``. Splits rows into ``num_shards`` contiguous
    chunks of a seeded permutation; sizes differ by at most 1.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    chunk_sizes = [n_rows // num_shards] * num_shards
    for i in range(n_rows % num_shards):
        chunk_sizes[i] += 1
    out = np.empty(n_rows, dtype=np.int64)
    cursor = 0
    for shard_idx, size in enumerate(chunk_sizes):
        shard_id = shard_idx + 1  # 1-indexed
        out[perm[cursor : cursor + size]] = shard_id
        cursor += size
    return out


def _stamp_row(row: dict, base_data_source: str, shard_id: int, original_row_index: int) -> dict:
    extra_info = dict(row.get("extra_info") or {})
    extra_info["prompt_uid"] = make_uid(base_data_source, original_row_index)
    extra_info["origin_shard"] = int(shard_id)
    extra_info["carry_history"] = []
    row["data_source"] = with_shard(base_data_source, shard_id)
    row["extra_info"] = extra_info
    return row


def _augmented_features(in_features: datasets.Features) -> datasets.Features:
    """Return an output Features that mirrors `in_features` but explicitly types
    the three new extra_info fields we add. The crucial one is
    ``carry_history``: an empty list gets inferred as ``list<null>`` by Arrow,
    which is technically valid but breaks downstream casts when stage k>1 tries
    to append real int stage indices. Pin it as ``Sequence(Value('int64'))``.
    """
    in_extra = in_features["extra_info"]
    # `extra_info` may surface as a plain dict (HuggingFace Features) — copy it
    # so we don't mutate the source dataset's features.
    out_extra = {k: v for k, v in in_extra.items()}
    out_extra["prompt_uid"]    = datasets.Value("int64")
    out_extra["origin_shard"]  = datasets.Value("int64")
    out_extra["carry_history"] = datasets.Sequence(datasets.Value("int64"))

    out = {k: v for k, v in in_features.items()}
    out["extra_info"] = out_extra
    return datasets.Features(out)


def _shard_one_source(
    src_path: str,
    base_data_source: str,
    raw_root: str,
    out_filename: str,
    seed: int,
) -> list[int]:
    """Read ``src_path``, split into NUM_SHARDS, write per-shard parquet files.

    Returns the per-shard row counts (length NUM_SHARDS).
    """
    ds = datasets.Dataset.from_parquet(src_path)
    n_rows = len(ds)
    shard_ids = _shard_assignments(n_rows, NUM_SHARDS, seed)
    out_features = _augmented_features(ds.features)

    counts = [0] * NUM_SHARDS
    for shard_idx in range(NUM_SHARDS):
        shard_id = shard_idx + 1
        keep_mask = (shard_ids == shard_id)
        keep_indices = np.nonzero(keep_mask)[0].tolist()
        sub = ds.select(keep_indices)

        # Stamp each row with uid (using the ORIGINAL row index, pre-permutation),
        # the shard suffix, etc.
        original_indices = keep_indices  # parallel to `sub`

        def map_fn(row, i, _ix=original_indices, _sid=shard_id):
            return _stamp_row(row, base_data_source, _sid, _ix[i])

        sub = sub.map(map_fn, with_indices=True, features=out_features)

        out_dir = ensure_dir(os.path.join(raw_root, f"shard_{shard_id}"))
        out_path = os.path.join(out_dir, out_filename)
        sub.to_parquet(out_path)
        counts[shard_idx] = len(sub)
        print(f"[shard_{shard_id}] {out_filename}: {len(sub)} rows -> {out_path}")

    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR,
                    help="Directory containing the per-source full parquets.")
    ap.add_argument("--chat-variant", choices=list(CHAT_VARIANTS), default="chat",
                    help=("Which chat parquet to shard: 'chat' uses "
                          "chat_train.parquet; 'chat_with_baseline' uses "
                          "chat_with_baseline_train.parquet."))
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the shard-assignment permutation.")
    args = ap.parse_args()

    train_dir = args.train_dir
    raw_root = ensure_dir(os.path.join(train_dir, "shards", "raw"))
    sources = _sources_for(args.chat_variant)
    print(f"[shard_train_data] chat variant: {args.chat_variant} ({CHAT_VARIANTS[args.chat_variant]})")

    summary: dict[str, list[int]] = {}
    for filename, base_ds in sources:
        src_path = os.path.join(train_dir, filename)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"missing source parquet: {src_path}")
        # Use a per-source seed offset so independent re-runs of one source are
        # reproducible without redoing the others identically.
        per_source_seed = args.seed + (hash(base_ds) & 0xFFFF)
        counts = _shard_one_source(src_path, base_ds, raw_root, filename, per_source_seed)
        summary[base_ds] = counts

        # Spot-check one row from shard_1 for sanity.
        sample_path = os.path.join(raw_root, "shard_1", filename)
        sample = pq.read_table(sample_path).slice(0, 1).to_pylist()[0]
        print(f"  sample data_source: {sample['data_source']!r}")
        print(f"  sample extra_info:  prompt_uid={sample['extra_info']['prompt_uid']} "
              f"origin_shard={sample['extra_info']['origin_shard']} "
              f"carry_history={sample['extra_info']['carry_history']}")

    # Materialize stage_1 = copy of raw/shard_1 (so the training script reads
    # from a uniform `shards/stage_{k}/` location for every stage).
    stage1_dir = ensure_dir(os.path.join(train_dir, "shards", "stage_1"))
    for filename, _ in sources:
        src = os.path.join(raw_root, "shard_1", filename)
        dst = os.path.join(stage1_dir, filename)
        shutil.copyfile(src, dst)
    print(f"[stage_1] copied raw/shard_1/* -> {stage1_dir}")

    # Print summary table.
    print("\n=== shard sizes ===")
    print(f"{'data_source':<35} " + " ".join(f"shard{i+1:>2}" for i in range(NUM_SHARDS)) + "    total")
    for ds, counts in summary.items():
        print(f"{ds:<35} " + " ".join(f"{c:>7}" for c in counts) + f"    {sum(counts):>5}")

    # Dump the summary as JSON for downstream tooling.
    summary_path = os.path.join(train_dir, "shards", "shard_sizes.json")
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in summary.items()}, f, indent=2)
    print(f"\nwrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
