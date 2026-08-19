"""Build curated per-source training subsets from a previous run's rollouts.

For each source dataset (ifeval / chat / safety / identity):
  - Keep ALL prompts whose mean rollout score < ``hard_threshold`` (default 0.6).
  - Sample prompts whose mean rollout score > ``easy_threshold`` (default 0.8)
    so they make up ``easy_fraction`` (default 0.15) of the output dataset.
  - Prompts in the middle band [hard_threshold, easy_threshold] are dropped.

Each output parquet preserves the EXACT schema of its source parquet — rows
are sliced from the source via ``pq.read_table().take([indices])`` so the
schema metadata, struct types, and column order are bit-identical to the
source. Suitable to drop directly into TRAIN_FILES.

The math for the easy-sample size:
    let H = number of hard prompts, E = easy prompts to keep, F = easy_fraction
    we want E / (H + E) = F   =>   E = (F / (1 - F)) * H

prompt_uid encoding (from full_mix.curriculum.encoding):
    prompt_uid = DATASET_ID[base] * UID_DATASET_STRIDE + row_index_in_source_parquet
The row index here is the row's position in the un-sharded source parquet
(set during preprocessing via ``with_indices=True``), so converting the
selected UIDs back to source rows is a single subtraction and a ``.take()``.

Usage:
    python -m full_mix.curriculum.build_curated_subset
    python -m full_mix.curriculum.build_curated_subset \\
        --rollout_dir /data/abdelrahman/verl/general_rollouts \\
        --output_dir  /data/abdelrahman/verl/data/full_mix/curated \\
        --hard_threshold 0.6 --easy_threshold 0.8 --easy_fraction 0.3
"""

import argparse
import json
import os
import random
from collections import defaultdict
from glob import glob
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from full_mix.curriculum.encoding import (
    DATASET_ID,
    UID_DATASET_STRIDE,
)


# Map base data_source -> source parquet filename. For chat we use the
# `chat_with_baseline` variant (the active one in train_fully_async_no_math.sh).
# Override at the CLI if you want chat_train.parquet instead — both have the
# same set of prompts at the same row indices.
SOURCE_PARQUETS: dict[str, str] = {
    "local/dolci-ifeval-32b":     "ifeval_train.parquet",
    "local/dolci-chat-32b":       "chat_with_baseline_train.parquet",
    "local/safety-dpo-reference": "safety_train.parquet",
    "local/avey-identity":        "identity_train.parquet",
}


def _iter_records(rollout_dir: str) -> Iterable[dict]:
    paths = sorted(glob(os.path.join(rollout_dir, "*.jsonl")))
    if not paths:
        raise SystemExit(f"no *.jsonl files under {rollout_dir}")
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def aggregate_per_prompt(rollout_dir: str) -> dict[int, dict[int, list[float]]]:
    """Returns {dataset_id: {prompt_uid: [score, score, ...]}}."""
    by_dataset: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    total = 0
    for rec in _iter_records(rollout_dir):
        if "prompt_uid" not in rec or "data_source_code" not in rec or "score" not in rec:
            continue
        ds_code = int(rec["data_source_code"])
        dataset_id = ds_code // 10  # tens digit (ones digit = shard or 0 if unsharded)
        prompt_uid = int(rec["prompt_uid"])
        by_dataset[dataset_id][prompt_uid].append(float(rec["score"]))
        total += 1
    print(f"  aggregated {total} rollouts across {sum(len(v) for v in by_dataset.values())} unique prompts")
    return by_dataset


def select_uids(
    prompt_scores: dict[int, list[float]],
    hard_threshold: float,
    easy_threshold: float,
    easy_fraction: float,
    rng: random.Random,
) -> tuple[list[int], dict[str, int | float]]:
    """Returns (sorted_selected_prompt_uids, stats_dict).

    Stats fields:
      total_in_run:        prompts observed in the rollouts for this source
      hard_count:          prompts with mean < hard_threshold (all kept)
      mid_count:           prompts in [hard_threshold, easy_threshold] (dropped)
      easy_count_available: prompts with mean > easy_threshold (pool to sample from)
      easy_count_sampled:  prompts actually drawn from the easy pool
      output_total:        hard_count + easy_count_sampled
      easy_fraction_actual: easy_count_sampled / output_total
    """
    means = {uid: float(np.mean(scores)) for uid, scores in prompt_scores.items()}
    hard = sorted(uid for uid, m in means.items() if m < hard_threshold)
    easy = sorted(uid for uid, m in means.items() if m > easy_threshold)
    mid = len(means) - len(hard) - len(easy)

    # Solve: E / (H + E) = F → E = (F / (1-F)) * H
    if easy_fraction >= 1.0 or easy_fraction < 0:
        raise SystemExit(f"easy_fraction must be in [0, 1); got {easy_fraction}")
    desired_easy = round(len(hard) * easy_fraction / (1 - easy_fraction))
    actual_easy = min(desired_easy, len(easy))

    if actual_easy < desired_easy:
        print(f"  WARNING: wanted {desired_easy} easy prompts, only {len(easy)} available "
              f"(easy_fraction will be lower than {easy_fraction:.2f})")

    easy_sample = rng.sample(easy, actual_easy) if actual_easy > 0 else []
    selected = sorted(hard + easy_sample)
    total = len(selected)
    return selected, {
        "total_in_run": len(means),
        "hard_count": len(hard),
        "mid_count": mid,
        "easy_count_available": len(easy),
        "easy_count_sampled": actual_easy,
        "output_total": total,
        "easy_fraction_actual": (actual_easy / total) if total else 0.0,
    }


def _promote_strings_to_large(schema: pa.Schema) -> pa.Schema:
    """Promote every top-level ``string`` column to ``large_string``.

    Reason: the medical_qa parquets ship with `data_source` / `ability` as
    ``large_string`` (Hazem's pipeline). When the curated full_mix parquets
    use plain ``string`` for those columns, HuggingFace Datasets refuses to
    concat the two: "data_source has unexpected type Value('string'),
    expected Value('large_string')". Promoting all top-level string columns
    is a safe superset — the medical schema is the dominant one we have to
    match, and ``string`` ⊂ ``large_string`` (size header only).
    """
    fields = []
    for f in schema:
        if pa.types.is_string(f.type):
            fields.append(pa.field(f.name, pa.large_string(), f.nullable, f.metadata))
        else:
            fields.append(f)
    return pa.schema(fields, metadata=schema.metadata)


def select_rows_from_parquet(
    parquet_path: str, prompt_uids: list[int], dataset_id: int
) -> pa.Table:
    """Slice the source parquet to the rows whose row_index == uid % STRIDE,
    and promote top-level ``string`` columns to ``large_string`` so the
    output can be concatenated with the medical_qa parquets at training time.
    """
    row_indices = [uid - dataset_id * UID_DATASET_STRIDE for uid in prompt_uids]
    t = pq.read_table(parquet_path)
    n_rows = t.num_rows
    valid = [i for i in row_indices if 0 <= i < n_rows]
    if len(valid) != len(row_indices):
        dropped = len(row_indices) - len(valid)
        print(f"  WARNING: dropping {dropped} UIDs whose row_index is out of range for "
              f"{os.path.basename(parquet_path)} (n_rows={n_rows})")
    sub = t.take(pa.array(valid, type=pa.int64()))
    promoted_schema = _promote_strings_to_large(sub.schema)
    if promoted_schema != sub.schema:
        sub = sub.cast(promoted_schema)
    return sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_dir",
                    default="/data/abdelrahman/verl/general_rollouts")
    ap.add_argument("--source_train_dir",
                    default="/data/abdelrahman/verl/data/full_mix/train")
    ap.add_argument("--output_dir",
                    default="/data/abdelrahman/verl/data/full_mix/curated")
    ap.add_argument("--hard_threshold", type=float, default=0.6,
                    help="Keep all prompts whose mean rollout score is BELOW this. Default 0.6.")
    ap.add_argument("--easy_threshold", type=float, default=0.8,
                    help="Sample from prompts whose mean rollout score is ABOVE this. Default 0.8.")
    ap.add_argument("--easy_fraction", type=float, default=0.15,
                    help="Target fraction of the OUTPUT that comes from the easy pool. Default 0.15.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for sampling the easy pool. Default 42.")
    ap.add_argument("--chat_variant", choices=["chat", "chat_with_baseline"],
                    default="chat_with_baseline",
                    help="Which chat parquet variant to use as source.")
    args = ap.parse_args()

    if args.chat_variant == "chat":
        SOURCE_PARQUETS["local/dolci-chat-32b"] = "chat_train.parquet"

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Reading rollouts from {args.rollout_dir} …")
    by_dataset = aggregate_per_prompt(args.rollout_dir)

    summary: dict[str, dict] = {}
    for base_ds, filename in SOURCE_PARQUETS.items():
        dataset_id = DATASET_ID[base_ds]
        prompt_scores = by_dataset.get(dataset_id, {})
        print(f"\n{base_ds}  (dataset_id={dataset_id}, source={filename})")
        if not prompt_scores:
            print("  no rollouts found — skipping")
            continue

        selected_uids, stats = select_uids(
            prompt_scores,
            hard_threshold=args.hard_threshold,
            easy_threshold=args.easy_threshold,
            easy_fraction=args.easy_fraction,
            rng=rng,
        )

        src = os.path.join(args.source_train_dir, filename)
        dst = os.path.join(args.output_dir, filename)
        sub = select_rows_from_parquet(src, selected_uids, dataset_id)
        pq.write_table(sub, dst)

        print(f"  total in rollouts        : {stats['total_in_run']}")
        print(f"  hard  (<{args.hard_threshold}): {stats['hard_count']}")
        print(f"  mid   [{args.hard_threshold}, {args.easy_threshold}]: {stats['mid_count']}  (dropped)")
        print(f"  easy  (>{args.easy_threshold}): {stats['easy_count_available']}  available, "
              f"{stats['easy_count_sampled']} sampled")
        print(f"  output rows written      : {sub.num_rows}")
        print(f"  actual easy fraction     : {stats['easy_fraction_actual']:.3f}")
        print(f"  wrote -> {dst}")

        summary[base_ds] = {
            "filename": filename,
            "source": src,
            "output": dst,
            **stats,
            "rows_written": sub.num_rows,
        }

    # Persist parameters + per-source counts for reproducibility.
    report_path = os.path.join(args.output_dir, "_build_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "rollout_dir": args.rollout_dir,
            "source_train_dir": args.source_train_dir,
            "chat_variant": args.chat_variant,
            "hard_threshold": args.hard_threshold,
            "easy_threshold": args.easy_threshold,
            "easy_fraction": args.easy_fraction,
            "seed": args.seed,
            "by_source": summary,
        }, f, indent=2)
    print(f"\nwrote report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
