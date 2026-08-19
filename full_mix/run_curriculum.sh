#!/usr/bin/env bash
# Drive the 4-stage curriculum end-to-end:
#   stage 1: train on shards/stage_1/ from base model
#   stage 2: build shards/stage_2/ from stage 1 dumps; train resuming stage_1 ckpt
#   stage 3: ditto, from stage 2
#   stage 4: ditto, from stage 3 (no transition after — final stage)
#
# Idempotent enough: each stage's training run uses verl's resume_path mode, so
# re-running this script picks up partial training. Re-running after a clean
# stage completion skips training (the next stage takes over) — we don't try
# to detect "stage already done" automatically; pass START_STAGE=k to skip.
#
# Run INSIDE the container started by dev/dev.sh, after running:
#   python -m full_mix.preprocess.shard_train_data
#
# Env knobs:
#   START_STAGE       (default 1) — first stage to run
#   END_STAGE         (default NUM_SHARDS=4) — last stage to run
#   PROJECT_NAME      (default RL-Exps) — wandb project; also drives ckpt path
#   CHECKPOINT_ROOT   (default /data/abdelrahman/verl/checkpoints/$PROJECT_NAME)
#   THRESHOLD         (default 0.875) — drop threshold for build_next_stage.py
#   DATA_DIR          (default /data/abdelrahman/verl/data/full_mix)
#   plus any of the env vars accepted by train_fully_async_no_math.sh
#     (TOTAL_ROLLOUT_STEPS, MAX_RESPONSE_LEN, NNODES_*, etc.)
#
# Anything in "$@" is forwarded verbatim to each training-script invocation.

set -euo pipefail

_FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(dirname "$_FM_DIR")"
TRAIN_SCRIPT="$_FM_DIR/train_fully_async_no_math.sh"

NUM_SHARDS=${NUM_SHARDS:-4}
START_STAGE=${START_STAGE:-1}
END_STAGE=${END_STAGE:-$NUM_SHARDS}
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${_REPO_ROOT}/checkpoints/${PROJECT_NAME}}
THRESHOLD=${THRESHOLD:-0.875}
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/full_mix}

export PROJECT_NAME

_latest_ckpt() {
  # Echo the absolute path to the highest-numbered global_step_* dir under
  # $1, or empty if none exists.
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  ls -d "$dir"/global_step_* 2>/dev/null \
    | awk -F'global_step_' '{print $NF, $0}' \
    | sort -n -k1,1 \
    | tail -n1 \
    | awk '{print $2}'
}

for (( STAGE=START_STAGE; STAGE<=END_STAGE; STAGE++ )); do
  echo
  echo "================================================================"
  echo "  curriculum stage $STAGE / $NUM_SHARDS"
  echo "================================================================"

  STAGE_DATA_DIR="$DATA_DIR/train/shards/stage_${STAGE}"
  if [[ ! -d "$STAGE_DATA_DIR" ]]; then
    if (( STAGE == 1 )); then
      echo "ERROR: $STAGE_DATA_DIR missing. Run:"
      echo "  python -m full_mix.preprocess.shard_train_data"
      exit 1
    fi
    PREV=$(( STAGE - 1 ))
    echo "[stage $STAGE] $STAGE_DATA_DIR missing — building from stage $PREV transition."
    python -m full_mix.curriculum.build_next_stage \
      --stage "$PREV" \
      --train_dir "$DATA_DIR/train" \
      --threshold "$THRESHOLD"
  fi

  # Resume path: stage 1 starts from base model (empty). Stages 2+ resume from
  # the latest checkpoint of the previous stage.
  RESUME_FROM_PATH=""
  if (( STAGE > 1 )); then
    PREV=$(( STAGE - 1 ))
    PREV_EXP="curriculum-stage${PREV}"
    PREV_CKPT_DIR="$CHECKPOINT_ROOT/$PREV_EXP"
    RESUME_FROM_PATH=$(_latest_ckpt "$PREV_CKPT_DIR" || true)
    if [[ -z "$RESUME_FROM_PATH" ]]; then
      echo "ERROR: no checkpoint found under $PREV_CKPT_DIR. Did stage $PREV finish?"
      exit 1
    fi
    echo "[stage $STAGE] resuming from $RESUME_FROM_PATH"
  else
    echo "[stage $STAGE] starting from base model (no RESUME_FROM_PATH)"
  fi

  # Hand off to the (parametrized) training script.
  STAGE="$STAGE" \
  RESUME_FROM_PATH="$RESUME_FROM_PATH" \
  PROJECT_NAME="$PROJECT_NAME" \
  bash "$TRAIN_SCRIPT" "$@"

  echo "[stage $STAGE] training complete."

  # Build the next stage's parquets, unless we just finished the last stage.
  if (( STAGE < NUM_SHARDS && STAGE < END_STAGE )); then
    NEXT=$(( STAGE + 1 ))
    NEXT_DIR="$DATA_DIR/train/shards/stage_${NEXT}"
    if [[ -d "$NEXT_DIR" ]]; then
      echo "[stage $STAGE -> $NEXT] $NEXT_DIR already exists; skipping transition build."
    else
      echo "[stage $STAGE -> $NEXT] building next stage parquets."
      python -m full_mix.curriculum.build_next_stage \
        --stage "$STAGE" \
        --train_dir "$DATA_DIR/train" \
        --threshold "$THRESHOLD"
    fi
  fi
done

echo
echo "curriculum done (stages $START_STAGE..$END_STAGE)"
