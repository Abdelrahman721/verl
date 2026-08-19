#!/usr/bin/env bash
# Reference: per-stage commands for the 4-stage curriculum.
# This file is NOT meant to be executed top-to-bottom. Copy-paste each block
# manually, in order, waiting for each training run / transition to finish
# before moving to the next.
#
# Conventions used below:
#   - Project name:        RL-Exps
#   - Repo root:           /data/abdelrahman/verl
#   - Checkpoint root:     /data/abdelrahman/verl/checkpoints/RL-Exps
#   - Per-stage exp name:  curriculum-stage{1,2,3,4}
#   - Per-stage dump dir:  ~/verl_dumps/curriculum_stage{1,2,3,4}_rollouts
#
# Run all commands from /data/abdelrahman/verl unless noted otherwise.
#
# CHAT VARIANT: there are two chat parquets in data/full_mix/train/ —
#   chat_train.parquet                   (--chat-variant chat,               default)
#   chat_with_baseline_train.parquet     (--chat-variant chat_with_baseline)
# Pick one and pass the SAME variant everywhere: shard_train_data.py,
# build_next_stage.py, and the training script (CHAT_VARIANT=...).
# Examples below default to plain "chat"; uncomment the with_baseline lines if
# you want the other variant.


# ---------------------------------------------------------------------------
# 0. one-time data prep — split each train parquet into 4 shards
# ---------------------------------------------------------------------------
# Writes:
#   data/full_mix/train/shards/raw/shard_{1..4}/{ifeval,<chat>,safety}_train.parquet
#   data/full_mix/train/shards/stage_1/{ifeval,<chat>,safety}_train.parquet
#   data/full_mix/train/shards/shard_sizes.json
# where <chat> is the parquet you picked via --chat-variant.

# python -m full_mix.preprocess.shard_train_data --chat-variant chat
python -m full_mix.preprocess.shard_train_data --chat-variant chat_with_baseline


# ---------------------------------------------------------------------------
# 1. stage 1 — train from base model on shards/stage_1/
# ---------------------------------------------------------------------------
# When STAGE is set, train_fully_async_no_math.sh defaults:
#   EXP_NAME           = curriculum-stage1
#   TRAIN_FILES        = data/full_mix/train/shards/stage_1/{ifeval,chat,safety}_train.parquet
#   ROLLOUT_DATA_DIR   = ~/verl_dumps/curriculum_stage1_rollouts
#   TOTAL_EPOCHS       = 1
#   RESUME_MODE        = disable  (because RESUME_FROM_PATH is empty)
#
# Override anything via env vars (TOTAL_ROLLOUT_STEPS, SAVE_FREQ, etc.).

# STAGE=1 CHAT_VARIANT=chat bash full_mix/train_fully_async_no_math.sh
STAGE=1 CHAT_VARIANT=chat_with_baseline bash full_mix/train_fully_async_no_math.sh


# After stage 1 finishes — build stage 2's training parquets.
# Reads rollouts from ~/verl_dumps/curriculum_stage1_rollouts/
# Writes data/full_mix/train/shards/stage_2/{ifeval,chat,safety}_train.parquet
#       and stage_2/_transition_report.json
# Drop threshold defaults to 0.875; pass --threshold to override.

# python -m full_mix.curriculum.build_next_stage --stage 1 --chat-variant chat
python -m full_mix.curriculum.build_next_stage --stage 1 --chat-variant chat_with_baseline


# ---------------------------------------------------------------------------
# Stage chaining via MODEL_PATH (NOT RESUME_FROM_PATH)
# ---------------------------------------------------------------------------
# Each stage starts as a FRESH training run from the previous stage's merged
# HF checkpoint. global_step resets to 0 every stage, optimizer/scheduler are
# re-initialized, and TOTAL_ROLLOUT_STEPS / TOTAL_EPOCHS map directly to
# "this stage's run length" (no surprise off-by-resumed-step).
#
# Two-step chain between stages:
#   1. Merge the previous stage's FSDP shards into HF safetensors.
#   2. Train the next stage with MODEL_PATH=<that merged dir>.


# ---------------------------------------------------------------------------
# 2. stage 2
# ---------------------------------------------------------------------------
# 2a. merge stage 1's last checkpoint to HF format.
PREV_CKPT=$(ls -d /data/abdelrahman/verl/checkpoints/RL-Exps/curriculum-stage1/global_step_* | sort -t_ -k3 -n | tail -1)
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$PREV_CKPT/actor" \
    --target_dir "$PREV_CKPT/merged_hf"

# 2b. train stage 2 from the merged base. NO RESUME_FROM_PATH.
STAGE=2 CHAT_VARIANT=chat_with_baseline \
MODEL_PATH="$PREV_CKPT/merged_hf" \
bash full_mix/train_fully_async_no_math.sh


# After stage 2 — build stage 3's training parquets.
python -m full_mix.curriculum.build_next_stage --stage 2 --chat-variant chat_with_baseline


# ---------------------------------------------------------------------------
# 3. stage 3
# ---------------------------------------------------------------------------
PREV_CKPT=$(ls -d /data/abdelrahman/verl/checkpoints/RL-Exps/curriculum-stage2/global_step_* | sort -t_ -k3 -n | tail -1)
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$PREV_CKPT/actor" \
    --target_dir "$PREV_CKPT/merged_hf"

STAGE=3 CHAT_VARIANT=chat_with_baseline \
MODEL_PATH="$PREV_CKPT/merged_hf" \
bash full_mix/train_fully_async_no_math.sh


# After stage 3 — build stage 4's training parquets.
python -m full_mix.curriculum.build_next_stage --stage 3 --chat-variant chat_with_baseline


# ---------------------------------------------------------------------------
# 4. stage 4 (final)
# ---------------------------------------------------------------------------
PREV_CKPT=$(ls -d /data/abdelrahman/verl/checkpoints/RL-Exps/curriculum-stage3/global_step_* | sort -t_ -k3 -n | tail -1)
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$PREV_CKPT/actor" \
    --target_dir "$PREV_CKPT/merged_hf"

STAGE=4 CHAT_VARIANT=chat_with_baseline \
MODEL_PATH="$PREV_CKPT/merged_hf" \
bash full_mix/train_fully_async_no_math.sh


# ---------------------------------------------------------------------------
# Sanity peeks (run any time after a stage finishes)
# ---------------------------------------------------------------------------

# Per-prompt mean score for stage k=1, sorted ascending (hardest first):
jq -s 'group_by(.prompt_uid) | map({prompt_uid: .[0].prompt_uid, n: length, mean: ([.[].score] | add / length)}) | sort_by(.mean)' \
  ~/verl_dumps/curriculum_stage1_rollouts/*.jsonl | head -40

# Per-shard distribution of the rows that ended up in stage 3's training set
# (each row's data_source preserves its ORIGIN shard, even after multiple carries):
python -c "
import datasets, collections
d = datasets.Dataset.from_parquet('data/full_mix/train/shards/stage_3/ifeval_train.parquet')
print(collections.Counter(r['data_source'] for r in d))
"

# Transition report from any stage (k -> k+1):
cat data/full_mix/train/shards/stage_2/_transition_report.json | jq .
