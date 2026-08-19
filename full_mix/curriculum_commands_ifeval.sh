#!/usr/bin/env bash
# Reference: per-stage commands for the IFEval-ONLY curriculum experiment.
# This file is NOT meant to be executed top-to-bottom — copy/paste each block
# in order, waiting for the previous training run / transition to finish.
#
# Conventions:
#   - Project name:      RL-Exps
#   - Repo root:         /data/abdelrahman/verl
#   - Checkpoint root:   /data/abdelrahman/verl/checkpoints/RL-Exps
#   - Per-stage exp:     ifeval-only-stage{1,2,3,4}
#   - Per-stage dump:    ~/verl_dumps/ifeval_only_stage{1,2,3,4}_rollouts
#   - Stage data dirs:   data/full_mix/train/shards/ifeval_only/stage_{1..4}/
#
# Reuses the existing raw shards (data/full_mix/train/shards/raw/shard_{1..4}/
# ifeval_train.parquet) — DO NOT re-run shard_train_data.py for this experiment;
# it would clobber the multi-source shards.
#
# Reward: full_mix/rewards/compute_score_ifeval.py routes IFEval rows to
# ifeval_quality_gated.compute_score, which multiplies the rule-based IFEval
# score by an LLM-judge "is this a genuine attempt vs gibberish/padding"
# multiplier (genuine=1.0, borderline=0.3, gibberish=0.0). Tuneable via env:
#   FULL_MIX_IFEVAL_GATE_ENABLED       (default 1)
#   FULL_MIX_IFEVAL_GATE_THRESHOLD     (default 0.5; skip judge below this)
#   FULL_MIX_IFEVAL_GATE_MULT_GENUINE  (default 1.0)
#   FULL_MIX_IFEVAL_GATE_MULT_BORDERLINE (default 0.3)
#   FULL_MIX_IFEVAL_GATE_MULT_GIBBERISH  (default 0.0)


# ---------------------------------------------------------------------------
# 0. one-time setup — seed stage_1 from the existing raw/shard_1
# ---------------------------------------------------------------------------
mkdir -p data/full_mix/train/shards/ifeval_only/stage_1
cp data/full_mix/train/shards/raw/shard_1/ifeval_train.parquet \
   data/full_mix/train/shards/ifeval_only/stage_1/ifeval_train.parquet


# ---------------------------------------------------------------------------
# 1. stage 1 — train from the base model on the IFEval shard 1
# ---------------------------------------------------------------------------
# When STAGE is set, train_fully_async_ifeval.sh defaults:
#   EXP_NAME           = ifeval-only-stage1
#   TRAIN_FILES        = data/full_mix/train/shards/ifeval_only/stage_1/ifeval_train.parquet
#   VAL_FILES          = data/full_mix/eval/ifeval_eval.parquet
#   ROLLOUT_DATA_DIR   = ~/verl_dumps/ifeval_only_stage1_rollouts
#   TOTAL_EPOCHS       = 1
#   REWARD_FN_PATH     = full_mix/rewards/compute_score_ifeval.py
# Override via env vars (TOTAL_ROLLOUT_STEPS, SAVE_FREQ, etc.).

STAGE=1 bash full_mix/train_fully_async_ifeval.sh


# After stage 1 finishes — build stage 2's training parquet.
# Reads rollouts from ~/verl_dumps/ifeval_only_stage1_rollouts/
# Writes data/full_mix/train/shards/ifeval_only/stage_2/ifeval_train.parquet
#       and stage_2/_transition_report.json
# Drop threshold defaults to 0.875; pass --threshold to override.

python -m full_mix.curriculum.build_next_stage_ifeval --stage 1


# ---------------------------------------------------------------------------
# 2. stage 2 — fresh run from stage 1's merged checkpoint
# ---------------------------------------------------------------------------
# 2a. merge stage 1's last checkpoint to HF format.
PREV_CKPT=$(ls -d /data/abdelrahman/verl/checkpoints/RL-Exps/ifeval-only-stage1/global_step_* | sort -t_ -k3 -n | tail -1)
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$PREV_CKPT/actor" \
    --target_dir "$PREV_CKPT/merged_hf"

# 2b. train stage 2 from the merged base. NO RESUME_FROM_PATH.
STAGE=2 \
MODEL_PATH="$PREV_CKPT/merged_hf" \
bash full_mix/train_fully_async_ifeval.sh

# After stage 2 — build stage 3's training parquet.
python -m full_mix.curriculum.build_next_stage_ifeval --stage 2


# ---------------------------------------------------------------------------
# 3. stage 3
# ---------------------------------------------------------------------------
PREV_CKPT=$(ls -d /data/abdelrahman/verl/checkpoints/RL-Exps/ifeval-only-stage2/global_step_* | sort -t_ -k3 -n | tail -1)
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$PREV_CKPT/actor" \
    --target_dir "$PREV_CKPT/merged_hf"

STAGE=3 \
MODEL_PATH="$PREV_CKPT/merged_hf" \
bash full_mix/train_fully_async_ifeval.sh

python -m full_mix.curriculum.build_next_stage_ifeval --stage 3


# ---------------------------------------------------------------------------
# 4. stage 4 (final)
# ---------------------------------------------------------------------------
PREV_CKPT=$(ls -d /data/abdelrahman/verl/checkpoints/RL-Exps/ifeval-only-stage3/global_step_* | sort -t_ -k3 -n | tail -1)
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$PREV_CKPT/actor" \
    --target_dir "$PREV_CKPT/merged_hf"

STAGE=4 \
MODEL_PATH="$PREV_CKPT/merged_hf" \
bash full_mix/train_fully_async_ifeval.sh


# ---------------------------------------------------------------------------
# Sanity peeks (after any stage finishes)
# ---------------------------------------------------------------------------

# Per-prompt mean score for stage 1, sorted ascending (hardest first):
jq -s 'group_by(.prompt_uid) | map({prompt_uid: .[0].prompt_uid, n: length, mean: ([.[].score] | add / length)}) | sort_by(.mean)' \
  ~/verl_dumps/ifeval_only_stage1_rollouts/*.jsonl | head -40

# 5%-bucket histogram for stage k=1 (uses the existing histogram script;
# data_source_code 11..14 will all decode to local/dolci-ifeval-32b):
python -m full_mix.curriculum.score_histogram \
  --rollout_dir ~/verl_dumps/ifeval_only_stage1_rollouts

# Transition report:
cat data/full_mix/train/shards/ifeval_only/stage_2/_transition_report.json | jq .
