"""IFEval-only curriculum stage transition.

Same algorithm as ``build_next_stage.py`` but operates on a single source
(IFEval) and reads/writes under ``shards/ifeval_only/stage_{k}/`` instead of
``shards/stage_{k}/``. Use this for the IFEval-only experiment so its
parquets and reports stay isolated from the multi-source curriculum.

Inputs (per stage k):
    --stage k                                                       (int 1..NUM_SHARDS-1)
    --rollout_dir   $HOME/verl_dumps/ifeval_only_stage{k}_rollouts/  (default)
    --shards_root   <train_dir>/shards/                              (default derived)
    --threshold     0.875                                            (default)

Outputs:
    <shards_root>/ifeval_only/stage_{k+1}/ifeval_train.parquet
    <shards_root>/ifeval_only/stage_{k+1}/_transition_report.json

Carry-forward semantics are unchanged from the multi-source pipeline:
classifies prompts by mean rollout score, drops the masters, retains the
origin-shard tag on `data_source`, appends `k` to `extra_info["carry_history"]`.
"""

import argparse
import json
import os
from collections import defaultdict
from glob import glob
from typing import Iterable

import datasets
import numpy as np

from full_mix.curriculum.encoding import (
    NUM_SHARDS,
    base_data_source,
)
from full_mix.preprocess._common import ensure_dir


DEFAULT_TRAIN_DIR = "/data/abdelrahman/verl/data/full_mix/train"
DEFAULT_DROP_THRESHOLD = 0.875

# Single source for this experiment.
IFEVAL_FILENAME = "ifeval_train.parquet"
IFEVAL_BASE_DATA_SOURCE = "local/dolci-ifeval-32b"


def _iter_jsonl_records(rollout_dir: str) -> Iterable[dict]:
    paths = sorted(glob(os.path.join(rollout_dir, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no *.jsonl files under {rollout_dir}")
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _aggregate_per_prompt(rollout_dir: str):
    """{prompt_uid: list[score]}. No shard sanity check — the dump can mix
    shards as carryovers retain their origin-shard tag."""
    per_prompt_scores: dict[int, list[float]] = defaultdict(list)
    total = 0
    for rec in _iter_jsonl_records(rollout_dir):
        prompt_uid = rec.get("prompt_uid")
        score = rec.get("score")
        if prompt_uid is None or score is None:
            raise ValueError(
                f"rollout record missing prompt_uid/score: keys={list(rec.keys())}"
            )
        per_prompt_scores[int(prompt_uid)].append(float(score))
        total += 1
    if not per_prompt_scores:
        raise RuntimeError(f"aggregated 0 valid records from {rollout_dir}")
    print(f"  aggregated {total} records across {len(per_prompt_scores)} unique prompts "
          f"from {rollout_dir}")
    return per_prompt_scores


def _classify(per_prompt_scores: dict[int, list[float]], threshold: float):
    mastered: set[int] = set()
    carried: set[int] = set()
    for prompt_uid, scores in per_prompt_scores.items():
        mean = float(np.mean(scores))
        if mean >= threshold:
            mastered.add(prompt_uid)
        else:
            carried.add(prompt_uid)
    return mastered, carried


def _load_stage_k_prompt_map(stage_k_dir: str, current_stage: int) -> dict[int, dict]:
    path = os.path.join(stage_k_dir, IFEVAL_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing stage parquet: {path}")
    ds = datasets.Dataset.from_parquet(path)
    prompt_to_row: dict[int, dict] = {}
    for row in ds:
        prompt_uid = int(row["extra_info"]["prompt_uid"])
        if prompt_uid in prompt_to_row:
            raise ValueError(f"duplicate prompt_uid {prompt_uid} in stage_{current_stage}")
        prompt_to_row[prompt_uid] = row
    return prompt_to_row


def _mark_carried_row(row: dict, current_stage: int) -> dict:
    extra = dict(row.get("extra_info") or {})
    history = list(extra.get("carry_history") or [])
    history.append(int(current_stage))
    extra["carry_history"] = history
    out = dict(row)
    out["extra_info"] = extra
    return out


def _promote_carry_history_features(features: datasets.Features) -> datasets.Features:
    """Force extra_info.carry_history -> Sequence(Value('int64')); see the
    multi-source build_next_stage for full rationale.
    """
    in_extra = features.get("extra_info")
    if in_extra is None:
        return features
    if not isinstance(in_extra, dict):
        in_extra = {k: v for k, v in in_extra.items()}
    out_extra = {k: v for k, v in in_extra.items()}
    out_extra["carry_history"] = datasets.Sequence(datasets.Value("int64"))
    out = {k: v for k, v in features.items()}
    out["extra_info"] = out_extra
    return datasets.Features(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True,
                    help="Current stage (1..NUM_SHARDS-1). Output is stage+1.")
    ap.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR)
    ap.add_argument("--rollout_dir", default=None,
                    help="Defaults to $HOME/verl_dumps/ifeval_only_stage{stage}_rollouts.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_DROP_THRESHOLD,
                    help="Drop prompts with mean rollout score >= threshold (default 0.875).")
    args = ap.parse_args()

    k = args.stage
    if not 1 <= k < NUM_SHARDS:
        raise SystemExit(f"--stage must be in [1, {NUM_SHARDS - 1}]; got {k}")
    next_k = k + 1

    rollout_dir = args.rollout_dir or os.path.expanduser(
        f"~/verl_dumps/ifeval_only_stage{k}_rollouts"
    )
    shards_root = os.path.join(args.train_dir, "shards")
    stage_k_dir = os.path.join(shards_root, "ifeval_only", f"stage_{k}")
    raw_next_dir = os.path.join(shards_root, "raw", f"shard_{next_k}")
    out_dir = ensure_dir(os.path.join(shards_root, "ifeval_only", f"stage_{next_k}"))

    print(f"[ifeval-only stage transition] {k} -> {next_k}")
    print(f"  rollout_dir   = {rollout_dir}")
    print(f"  stage_{k}_dir = {stage_k_dir}")
    print(f"  raw_next_dir  = {raw_next_dir}")
    print(f"  out_dir       = {out_dir}")
    print(f"  threshold     = {args.threshold}")

    per_prompt_scores = _aggregate_per_prompt(rollout_dir)
    mastered, carried = _classify(per_prompt_scores, args.threshold)
    print(f"  classified: mastered={len(mastered)}  carried={len(carried)}")

    prompt_to_row = _load_stage_k_prompt_map(stage_k_dir, current_stage=k)
    print(f"  stage_{k} rows total: {len(prompt_to_row)}")

    stage_k_prompts = set(prompt_to_row.keys())
    seen_prompts = set(per_prompt_scores.keys())
    unseen = stage_k_prompts - seen_prompts
    if unseen:
        print(f"  WARNING: {len(unseen)} stage_{k} prompts missing from rollouts; carrying them.")
        carried |= unseen
    extraneous = seen_prompts - stage_k_prompts
    if extraneous:
        print(f"  WARNING: {len(extraneous)} prompts in dumps not in stage_{k}; ignoring.")
        carried -= extraneous
        mastered -= extraneous

    carried_rows = [
        _mark_carried_row(prompt_to_row[uid], current_stage=k) for uid in carried
    ]

    raw_path = os.path.join(raw_next_dir, IFEVAL_FILENAME)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"missing raw shard parquet: {raw_path}")
    raw_ds = datasets.Dataset.from_parquet(raw_path)
    target_features = _promote_carry_history_features(raw_ds.features)
    if raw_ds.features != target_features:
        raw_ds = raw_ds.cast(target_features)

    if carried_rows:
        carried_ds = datasets.Dataset.from_list(carried_rows, features=target_features)
        combined = datasets.concatenate_datasets([raw_ds, carried_ds])
    else:
        combined = raw_ds

    out_path = os.path.join(out_dir, IFEVAL_FILENAME)
    combined.to_parquet(out_path)
    print(f"  wrote {out_path}: raw={len(raw_ds)} + carried={len(carried_rows)} "
          f"= total={len(combined)}")

    # Origin-shard distribution within carried rows (informational).
    carried_origin_counts: dict[str, int] = defaultdict(int)
    for row in carried_rows:
        carried_origin_counts[row["data_source"]] += 1

    report = {
        "stage": k,
        "next_stage": next_k,
        "threshold": args.threshold,
        "rollout_dir": rollout_dir,
        "stage_k_total": len(stage_k_prompts),
        "mastered_total": len(mastered),
        "carried_total": len(carried),
        "carried_origin_shard_distribution": dict(carried_origin_counts),
        "next_stage_file": {
            "path": out_path,
            "raw": len(raw_ds),
            "carried": len(carried_rows),
            "total": len(combined),
        },
    }
    report_path = os.path.join(out_dir, "_transition_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote transition report -> {report_path}")


if __name__ == "__main__":
    main()
