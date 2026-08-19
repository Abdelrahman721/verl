"""Curriculum stage transition.

Inputs (per stage k):
    --stage k                                                       (int 1..NUM_SHARDS-1)
    --rollout_dir   $HOME/verl_dumps/curriculum_stage{k}_rollouts/  (default)
    --shards_root   <train_dir>/shards/                             (default derived)
    --threshold     0.875                                            (default)

Outputs:
    <shards_root>/stage_{k+1}/{ifeval,chat,safety}_train.parquet
    <shards_root>/stage_{k+1}/_transition_report.json

Algorithm:
1. Read every JSONL under rollout_dir. Each record carries int `prompt_uid`,
   int `data_source` (encoded), float `score`. Records may come from multiple
   shards in the same stage (e.g. stage 3 includes carryovers originally from
   shards 1 and 2) — that's fine, no sanity check on the shard digit.
2. Group by prompt_uid. mean_score = mean(scores). If mean >= threshold ->
   mastered (drop). Else -> carry. prompt_uids absent from the dump default to
   carry (defensive).
3. Load <shards_root>/stage_{k}/*.parquet. Build prompt_uid -> row map.
4. For each carried prompt_uid, append k to extra_info["carry_history"]. Leave
   `data_source` untouched so the row keeps its origin-shard tag through every
   carry — a prompt that started in shard 1 and was carried into stage 4 still
   shows up in metrics/dumps as `...#shard1`.
5. Load <shards_root>/raw/shard_{k+1}/*.parquet (tagged with #shard{k+1}).
6. For each base data_source, write
   <shards_root>/stage_{k+1}/{base}_train.parquet = concat(raw_shard_{k+1}, carried).
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
from full_mix.preprocess.shard_train_data import CHAT_VARIANTS, _sources_for


DEFAULT_TRAIN_DIR = "/data/abdelrahman/verl/data/full_mix/train"
DEFAULT_DROP_THRESHOLD = 0.875


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
    """Returns per_prompt_scores: dict[int, list[float]] keyed by prompt_uid.

    A stage's dump can legitimately contain records from multiple shards (the
    raw shard plus any prompts carried from earlier stages, each retaining its
    origin-shard tag), so we don't check the shard digit.
    """
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


def _load_stage_k_prompt_map(
    stage_k_dir: str, current_stage: int, sources: list[tuple[str, str]]
) -> dict[int, dict]:
    """prompt_uid -> single-row dict (the parquet row)."""
    prompt_to_row: dict[int, dict] = {}
    for filename, _base in sources:
        path = os.path.join(stage_k_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing stage parquet: {path}")
        ds = datasets.Dataset.from_parquet(path)
        for row in ds:
            prompt_uid = int(row["extra_info"]["prompt_uid"])
            if prompt_uid in prompt_to_row:
                raise ValueError(f"duplicate prompt_uid {prompt_uid} in stage_{current_stage}")
            prompt_to_row[prompt_uid] = row
    return prompt_to_row


def _mark_carried_row(row: dict, current_stage: int) -> dict:
    """Append `current_stage` to `extra_info["carry_history"]`. Leave
    `data_source` untouched so the row preserves its origin-shard tag."""
    extra = dict(row.get("extra_info") or {})
    history = list(extra.get("carry_history") or [])
    history.append(int(current_stage))
    extra["carry_history"] = history
    out = dict(row)
    out["extra_info"] = extra
    return out


def _promote_carry_history_features(features: datasets.Features) -> datasets.Features:
    """Force ``extra_info.carry_history`` to ``Sequence(Value('int64'))``.

    Older raw/shard parquets (written before the typed-features fix in
    shard_train_data.py) carry ``list<null>`` — Arrow's default for an empty
    list. When we concat them with carried rows that hold real int stage
    indices, the cast fails. Promote both sides to the int64 schema before
    concatenating; empty list<null> casts cleanly to empty list<int64>.
    """
    in_extra = features.get("extra_info")
    if in_extra is None:
        return features
    if not isinstance(in_extra, dict):
        # Defensive: in some Datasets versions struct features expose as a
        # Features instance, which behaves like a dict.
        in_extra = {k: v for k, v in in_extra.items()}
    out_extra = {k: v for k, v in in_extra.items()}
    out_extra["carry_history"] = datasets.Sequence(datasets.Value("int64"))
    out = {k: v for k, v in features.items()}
    out["extra_info"] = out_extra
    return datasets.Features(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True,
                    help="Current stage index (1..NUM_SHARDS-1). Output stage is stage+1.")
    ap.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR)
    ap.add_argument("--rollout_dir", default=None,
                    help="Defaults to $HOME/verl_dumps/curriculum_stage{stage}_rollouts.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_DROP_THRESHOLD,
                    help="Drop uids with mean rollout score >= threshold. Default 0.875.")
    ap.add_argument("--chat-variant", choices=list(CHAT_VARIANTS), default="chat",
                    help=("Chat parquet variant the curriculum was sharded with. "
                          "Must match the value passed to shard_train_data.py."))
    args = ap.parse_args()
    sources = _sources_for(args.chat_variant)

    k = args.stage
    if not 1 <= k < NUM_SHARDS:
        raise SystemExit(f"--stage must be in [1, {NUM_SHARDS - 1}]; got {k}")
    next_k = k + 1

    rollout_dir = args.rollout_dir or os.path.expanduser(
        f"~/verl_dumps/curriculum_stage{k}_rollouts"
    )
    shards_root = os.path.join(args.train_dir, "shards")
    stage_k_dir = os.path.join(shards_root, f"stage_{k}")
    raw_next_dir = os.path.join(shards_root, "raw", f"shard_{next_k}")
    out_dir = ensure_dir(os.path.join(shards_root, f"stage_{next_k}"))

    print(f"[stage transition] {k} -> {next_k}")
    print(f"  rollout_dir = {rollout_dir}")
    print(f"  stage_{k}_dir = {stage_k_dir}")
    print(f"  raw_next_dir  = {raw_next_dir}")
    print(f"  out_dir       = {out_dir}")
    print(f"  threshold     = {args.threshold}")

    # 1+2: aggregate per-prompt scores, classify.
    per_prompt_scores = _aggregate_per_prompt(rollout_dir)
    mastered, carried = _classify(per_prompt_scores, args.threshold)
    print(f"  classified: mastered={len(mastered)}  carried={len(carried)}")

    # 3+4: load stage_{k} rows and mark the carried ones.
    prompt_to_row = _load_stage_k_prompt_map(stage_k_dir, current_stage=k, sources=sources)
    print(f"  stage_{k} rows total: {len(prompt_to_row)}")

    # Defensive: any prompt_uid that participated in stage_k but is missing
    # from the dump (e.g. the run died early) is treated as carry.
    stage_k_prompts = set(prompt_to_row.keys())
    seen_prompts = set(per_prompt_scores.keys())
    unseen = stage_k_prompts - seen_prompts
    if unseen:
        print(f"  WARNING: {len(unseen)} stage_{k} prompts missing from rollouts; carrying them.")
        carried |= unseen
    # Symmetric defense: any prompt_uid in dumps but not in stage_k (shouldn't happen) is dropped.
    extraneous = seen_prompts - stage_k_prompts
    if extraneous:
        print(f"  WARNING: {len(extraneous)} prompts in dumps not in stage_{k}; ignoring.")
        carried -= extraneous
        mastered -= extraneous

    # Bucket carried rows by base data_source. Origin-shard tag is preserved.
    carried_by_base: dict[str, list[dict]] = defaultdict(list)
    for prompt_uid in carried:
        row = prompt_to_row[prompt_uid]
        marked = _mark_carried_row(row, current_stage=k)
        base = base_data_source(marked["data_source"])
        carried_by_base[base].append(marked)

    # 5+6: union with raw/shard_{k+1}.
    summary: dict[str, dict[str, int]] = {}
    for filename, base_ds in sources:
        raw_path = os.path.join(raw_next_dir, filename)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"missing raw shard parquet: {raw_path}")
        raw_ds = datasets.Dataset.from_parquet(raw_path)

        # Promote carry_history to a typed int64 list so old (list<null>) raw
        # shards can concat with the freshly-marked carry rows without an
        # int64 -> null cast failure.
        target_features = _promote_carry_history_features(raw_ds.features)
        if raw_ds.features != target_features:
            raw_ds = raw_ds.cast(target_features)

        carried_rows = carried_by_base.get(base_ds, [])
        if carried_rows:
            carried_ds = datasets.Dataset.from_list(carried_rows, features=target_features)
            combined = datasets.concatenate_datasets([raw_ds, carried_ds])
        else:
            combined = raw_ds

        out_path = os.path.join(out_dir, filename)
        combined.to_parquet(out_path)
        summary[base_ds] = {
            "raw": len(raw_ds),
            "carried": len(carried_rows),
            "total": len(combined),
        }
        print(f"  wrote {out_path}: raw={len(raw_ds)} + carried={len(carried_rows)} "
              f"= total={len(combined)}")

    # Per-source stats over the just-finished stage.
    per_source_stats: dict[str, dict] = defaultdict(lambda: {"mastered": 0, "carried": 0, "total": 0})
    for prompt_uid in stage_k_prompts:
        row = prompt_to_row[prompt_uid]
        base = base_data_source(row["data_source"])
        per_source_stats[base]["total"] += 1
        if prompt_uid in mastered:
            per_source_stats[base]["mastered"] += 1
        else:
            per_source_stats[base]["carried"] += 1

    report = {
        "stage": k,
        "next_stage": next_k,
        "threshold": args.threshold,
        "rollout_dir": rollout_dir,
        "stage_k_total": len(stage_k_prompts),
        "mastered_total": len(mastered),
        "carried_total": len(carried),
        "per_source_classification": dict(per_source_stats),
        "next_stage_files": summary,
    }
    report_path = os.path.join(out_dir, "_transition_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote transition report -> {report_path}")

    print("\n=== per-source classification (stage {}) ===".format(k))
    for base, stats in per_source_stats.items():
        print(f"  {base}: total={stats['total']} mastered={stats['mastered']} carried={stats['carried']}")


if __name__ == "__main__":
    main()
