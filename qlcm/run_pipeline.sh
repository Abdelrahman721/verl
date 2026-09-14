#!/usr/bin/env bash
# Drive the whole QLCM GRPO pipeline end-to-end.
#
#   general → 1 → 2 → 3 → 4 → 5 → 6 → 8
#
# Each stage: build its training data from the PREVIOUS stage's rollout dumps,
# train, then merge the resulting FSDP checkpoint to HuggingFace format so the
# next stage can load it as `actor_rollout_ref.model.path`.
#
# The stage numbering deliberately skips 7, 9 and 10 — see README.md. Stage 8
# is the FINAL stage; its merged checkpoint is the pipeline's output.
#
# Run INSIDE the container started by dev/dev.sh, on the Ray head node, after
#   bash qlcm/preprocess_all.sh
# has produced the general train/eval parquets.
#
# Env knobs:
#   START_STAGE / END_STAGE  first / last stage to run (default general / 8)
#   PROJECT_NAME             wandb project + checkpoint dir (default QLCM)
#   CHECKPOINT_ROOT          default /data/abdelrahman/verl/checkpoints/$PROJECT_NAME
#   BASE_MODEL               CPT+SFT checkpoint the `general` stage starts from
#   INIT_MODEL               pin the model START_STAGE begins from, instead of
#                            auto-resolving the previous stage's LATEST
#                            checkpoint. Accepts a global_step_N dir (merged on
#                            demand) or an already-merged HF dir. Use it to
#                            branch from a checkpoint that is not the newest —
#                            e.g. you stopped a stage early and want step 260
#                            when step 300 also exists.
#   DATA_DIR                 derived data root (default .../data/qlcm)
#   MEDICAL_QA_SRC           shared raw medical corpus (read-only)
#   BENCH_PARQUET            medical benchmark parquet — REQUIRED for stage 3
#   CARVE_PARQUETS           optional space-separated carve parquets for stage 2
#   DUMPS_DIR                rollout dump root (default /data/abdelrahman/verl/dumps/qlcm)
#   THRESHOLD                mastery cut-off for the curriculum (default 0.6)
#   SKIP_MERGE=1             don't merge checkpoints (you'll do it by hand)
#   MERGE_56=1               run stages 5+6 as ONE merged stage (id 56)
#   SKIP_STAGE2=1            stage 3 draws BOTH replay pools from stage 1
#                            (auto-detected when stage-2 rollouts are absent)
#   DRY_RUN=1                print what would happen, run nothing
#
# Anything in "$@" is forwarded verbatim to each training script, so extra
# Hydra overrides work:  bash qlcm/run_pipeline.sh trainer.test_freq=5

set -euo pipefail

_QLCM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(dirname "$_QLCM_DIR")"
cd "$_REPO_ROOT"

# Single source of truth for environment configuration.
# shellcheck source=load_runtime_env.sh
source "$_QLCM_DIR/load_runtime_env.sh"

PROJECT_NAME=${PROJECT_NAME:-QLCM}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${_REPO_ROOT}/checkpoints/${PROJECT_NAME}}
BASE_MODEL=${BASE_MODEL:-/data/abdelrahman/qwen-sft/qlcm/out-qwen3-1.7b/coding-mixed-sft/final}
INIT_MODEL=${INIT_MODEL:-}
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/qlcm}
MEDICAL_QA_SRC=${MEDICAL_QA_SRC:-/data/abdelrahman/verl/data/medical_qa}
DUMPS_DIR=${DUMPS_DIR:-/data/abdelrahman/verl/dumps/qlcm}
THRESHOLD=${THRESHOLD:-0.6}
BENCH_PARQUET=${BENCH_PARQUET:-}
CARVE_PARQUETS=${CARVE_PARQUETS:-}
SKIP_MERGE=${SKIP_MERGE:-0}
DRY_RUN=${DRY_RUN:-0}
PY=${PYTHON:-python3}

# MERGE_56=1 replaces the separate 5 and 6 stages with one merged run.
if [[ "${MERGE_56:-0}" == "1" ]]; then
  STAGES=(general 1 2 3 4 56 8)
else
  STAGES=(general 1 2 3 4 5 6 8)
fi
START_STAGE=${START_STAGE:-general}
END_STAGE=${END_STAGE:-8}

# A START/END stage that is not in the chain must fail loudly. The main loop
# simply matches nothing otherwise, so the run "succeeds" in seconds having
# trained nothing — which is how a dropped MERGE_56 used to present.
for _s in "$START_STAGE" "$END_STAGE"; do
  if [[ ! " ${STAGES[*]} " == *" $_s "* ]]; then
    echo "ERROR: stage '$_s' is not in this chain: ${STAGES[*]}" >&2
    [[ "$_s" == "56" ]] && echo "       stage 56 is the merged 5+6 run — it needs MERGE_56=1" >&2
    exit 2
  fi
done

# Curriculum pool budgets. Defaults reproduce the 32B run; scale them down for
# a smaller actor or a shorter schedule.
STAGE2_CONV_KEEP_N=${STAGE2_CONV_KEEP_N:-5000}
STAGE3_STAGE1_MASTERED_N=${STAGE3_STAGE1_MASTERED_N:-1000}
STAGE3_STAGE2_UNMASTERED_N=${STAGE3_STAGE2_UNMASTERED_N:-2000}
STAGE3_STAGE2_MASTERED_N=${STAGE3_STAGE2_MASTERED_N:-1000}
STAGE3_BENCH_MCQ_N=${STAGE3_BENCH_MCQ_N:-8507}
STAGE3_BENCH_NUMERIC_N=${STAGE3_BENCH_NUMERIC_N:-1995}
STAGE3_BENCH_MEDEC_N=${STAGE3_BENCH_MEDEC_N:-435}
STAGE3_BENCH_TEXT_N=${STAGE3_BENCH_TEXT_N:-6256}


# ===========================================================================
# Helpers
# ===========================================================================

# Progress goes to stderr: _stage_model / _merge_ckpt return paths on stdout and
# are read with $(...), so anything chatty on stdout would corrupt them.
log() { echo -e "\n[qlcm] $*" >&2; }

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    { printf '  DRY: '; printf '%q ' "$@"; printf '\n'; } >&2
  else
    "$@"
  fi
}

_exp_name() {
  case "$1" in
    general) echo "general-phase" ;;
    1)       echo "medical-qa-fresh" ;;
    2)       echo "medical-qa-stage2" ;;
    3)       echo "medical-qa-stage3" ;;
    4)       echo "medical-qa-stage4-coding" ;;
    5)       echo "medical-qa-stage5-coding" ;;
    6)       echo "medical-qa-stage6-coding" ;;
    56)      echo "medical-qa-stage56-coding" ;;   # merged 5+6
    8)       echo "medical-qa-stage8-coding" ;;
    *)       echo "unknown-stage-$1" ;;
  esac
}

_train_script() {
  case "$1" in
    general) echo "$_QLCM_DIR/train_general.sh" ;;
    *)       echo "$_QLCM_DIR/train_medical_qa_stage$1.sh" ;;
  esac
}

_rollout_dir() {
  case "$1" in
    general) echo "$DUMPS_DIR/general_rollouts" ;;
    1)       echo "$DUMPS_DIR/medical_qa_rollouts" ;;
    *)       echo "$DUMPS_DIR/medical_qa_stage$1_rollouts" ;;
  esac
}

_stage_data_dir() {
  case "$1" in
    general) echo "$DATA_DIR/train" ;;
    1)       echo "$MEDICAL_QA_SRC" ;;
    *)       echo "$DATA_DIR/medical_qa_stage$1" ;;
  esac
}

# Highest-numbered global_step_* dir under $1, or empty.
_latest_ckpt() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  ls -d "$dir"/global_step_* 2>/dev/null \
    | awk -F'global_step_' '{print $NF, $0}' \
    | sort -n -k1,1 | tail -n1 | awk '{print $2}'
}

# Merge the FSDP shards of a checkpoint dir into merged_hf_model/.
_merge_ckpt() {
  local ckpt="$1"
  local target="$ckpt/merged_hf_model"
  if [[ -f "$target/config.json" ]]; then
    echo "$target"; return 0
  fi
  if [[ "$SKIP_MERGE" == "1" ]]; then
    echo "SKIP_MERGE=1 but $target does not exist" >&2; return 1
  fi
  log "merging $ckpt/actor -> $target"
  # The >&2 is load-bearing. This function RETURNS the target path on stdout and
  # is always called via $(...). verl.model_merger prints its config dump, device
  # mesh and save progress to stdout, so without the redirect every one of those
  # lines is captured as part of the "path" and handed to the next tool — which
  # is how a merge inside the pipeline ended up passing a multi-line blob as
  # --tokenizer. Earlier merges were all done by hand first, so this function
  # only ever hit the early return above and the bug stayed hidden.
  run $PY -m verl.model_merger merge \
      --backend fsdp --local_dir "$ckpt/actor" --target_dir "$target" >&2
  if [[ "$DRY_RUN" != "1" && ! -f "$target/config.json" ]]; then
    echo "ERROR: merge produced no config.json at $target" >&2
    return 1
  fi
  echo "$target"
}

# Tokenizer for the stats_* scripts: it must match the chat template the
# rollouts were generated with. Normally that is the previous stage's model,
# but when START_STAGE is pinned with INIT_MODEL the pinned checkpoint IS the
# previous stage's model — and resolving _stage_model would instead merge that
# stage's LATEST checkpoint, which is both the wrong target and a needless
# ~20GB merge when you stopped the stage early.
_tokenizer_for() {
  local prev="$1"
  if [[ -n "$INIT_MODEL" ]]; then
    if [[ -f "$INIT_MODEL/config.json" ]]; then echo "$INIT_MODEL"; return 0; fi
    if [[ -d "$INIT_MODEL/actor" ]]; then _merge_ckpt "$INIT_MODEL"; return 0; fi
  fi
  _stage_model "$prev"
}

# HF-format model produced by stage $1 (merging it first if needed).
_stage_model() {
  local stage="$1"
  local exp; exp="$(_exp_name "$stage")"
  local ckpt; ckpt="$(_latest_ckpt "$CHECKPOINT_ROOT/$exp")"
  if [[ -z "$ckpt" ]]; then
    echo "ERROR: no checkpoint under $CHECKPOINT_ROOT/$exp — did stage $stage finish?" >&2
    return 1
  fi
  _merge_ckpt "$ckpt"
}

_require() {
  local what="$1"; shift
  for f in "$@"; do
    if [[ ! -e "$f" ]]; then
      echo "ERROR: $what needs $f, which does not exist." >&2
      return 1
    fi
  done
}

_have_rollouts() {
  local dir="$1"
  [[ -d "$dir" ]] && compgen -G "$dir"/*.jsonl > /dev/null
}


# ===========================================================================
# Per-stage data builds — each reads the PREVIOUS stage's rollout dumps
# ===========================================================================

build_data_for() {
  local stage="$1"

  case "$stage" in

  general)
    _require "the general stage" \
      "$DATA_DIR/train/ifeval_train.parquet" \
      "$DATA_DIR/train/chat_with_baseline_train.parquet" \
      "$DATA_DIR/train/safety_train.parquet" \
      "$DATA_DIR/train/identity_train.parquet"
    log "general stage data already preprocessed"
    ;;

  1)
    # Retention for the medical stage = difficulty-curated subset of the
    # general train sets, scored by the general stage's own rollouts.
    if [[ -f "$DATA_DIR/curated/ifeval_train.parquet" ]]; then
      log "stage 1: curated retention already built"
    else
      local rd; rd="$(_rollout_dir general)"
      _have_rollouts "$rd" || { echo "ERROR: no rollout dumps in $rd" >&2; return 1; }
      log "stage 1: building curated retention from general rollouts"
      run $PY -m qlcm.curriculum.build_curated_subset \
        --rollout_dir "$rd" \
        --source_train_dir "$DATA_DIR/train" \
        --output_dir "$DATA_DIR/curated" \
        --hard_threshold "$THRESHOLD" \
        --easy_threshold 0.8 --easy_fraction 0.2 --seed 42 \
        --chat_variant chat_with_baseline
    fi
    _require "stage 1" "$MEDICAL_QA_SRC/train.parquet" "$MEDICAL_QA_SRC/val.parquet"
    ;;

  2)
    local out="$DATA_DIR/medical_qa_stage2"
    [[ -f "$out/train.parquet" ]] && { log "stage 2 data already built"; return 0; }
    local rd; rd="$(_rollout_dir 1)"
    _have_rollouts "$rd" || { echo "ERROR: no rollout dumps in $rd" >&2; return 1; }
    local tok; tok="$(_tokenizer_for 1)"
    local carve_args=()
    for c in $CARVE_PARQUETS; do carve_args+=(--carve "$c"); done
    log "stage 2: scoring stage-1 rollouts"
    run mkdir -p "$out"
    run $PY -m qlcm.curriculum.stats_medical_qa_next \
      --rollout_dir "$rd" --source "$MEDICAL_QA_SRC/train.parquet" \
      --tokenizer "$tok" --out "$out/_stats_cache.json" "${carve_args[@]}"
    log "stage 2: building curriculum parquet"
    run $PY -m qlcm.curriculum.build_medical_qa_next \
      --cache "$out/_stats_cache.json" --threshold "$THRESHOLD" \
      --conv_keep_n "$STAGE2_CONV_KEEP_N" --output "$out" --seed 1234
    ;;

  3)
    local out="$DATA_DIR/medical_qa_stage3"
    [[ -f "$out/train.parquet" ]] && { log "stage 3 data already built"; return 0; }
    [[ -n "$BENCH_PARQUET" ]] || {
      echo "ERROR: stage 3 folds in the medical benchmark pool — set BENCH_PARQUET." >&2; return 1; }
    _require "stage 3" "$BENCH_PARQUET"
      local rd1 rd2 s2src tokstage a2m
      rd1="$(_rollout_dir 1)"; rd2="$(_rollout_dir 2)"
      s2src="$DATA_DIR/medical_qa_stage2/train.parquet"
      tokstage=2
      a2m="$STAGE3_STAGE2_MASTERED_N"

      # Stage 2 may be skipped. The stage-3 builder wants TWO scored stages, so
      # alias the stage-2 inputs onto stage 1: the `stage1` and `stage2` cache
      # blocks then hold the same per-prompt scores, A1m draws from stage-1
      # mastered and A2u from stage-1 unmastered (disjoint sets, so no dupes),
      # and the unified extra_info type is unchanged because both sources are
      # the same medical_qa_compat parquet. A2m is forced to 0 — with one shared
      # stage it would resample the mastered pool A1m already covers.
      if [[ "${SKIP_STAGE2:-0}" == "1" ]] || ! _have_rollouts "$rd2"; then
        log "stage 3: no stage-2 rollouts — aliasing stage-2 inputs to stage 1, A2m=0"
        rd2="$rd1"; s2src="$MEDICAL_QA_SRC/train.parquet"; tokstage=1; a2m=0
      fi
      _have_rollouts "$rd1" || { echo "ERROR: no rollout dumps in $rd1" >&2; return 1; }
      local tok; tok="$(_tokenizer_for $tokstage)"
    log "stage 3: scoring stage-1 + stage-2 rollouts against the bench pool"
    run mkdir -p "$out"
    run $PY -m qlcm.curriculum.stats_medical_qa_stage3 \
      --stage1_rollout_dir "$rd1" --stage1_source "$MEDICAL_QA_SRC/train.parquet" \
      --stage2_rollout_dir "$rd2" --stage2_source "$s2src" \
      --tokenizer "$tok" --bench "$BENCH_PARQUET" --out "$out/_stats_cache.json"
    log "stage 3: building replay + benchmark parquet"
    run $PY -m qlcm.curriculum.build_medical_qa_stage3 \
      --cache "$out/_stats_cache.json" --threshold "$THRESHOLD" \
      --stage1_mastered_n   "$STAGE3_STAGE1_MASTERED_N" \
      --stage2_unmastered_n "$STAGE3_STAGE2_UNMASTERED_N" \
      --stage2_mastered_n   "$a2m" \
      --bench_mcq_n         "$STAGE3_BENCH_MCQ_N" \
      --bench_numeric_n     "$STAGE3_BENCH_NUMERIC_N" \
      --bench_medec_n       "$STAGE3_BENCH_MEDEC_N" \
      --bench_text_n        "$STAGE3_BENCH_TEXT_N" \
      --output "$out" --seed 1234
    ;;

  4)
    local out="$DATA_DIR/medical_qa_stage4"
    [[ -f "$out/train.parquet" ]] && { log "stage 4 data already built"; return 0; }
    local rd; rd="$(_rollout_dir 3)"
    _have_rollouts "$rd" || { echo "ERROR: no rollout dumps in $rd" >&2; return 1; }
    local tok; tok="$(_tokenizer_for 3)"
    log "stage 4: building coding-only train (SNOMED single-label + ICD multi-label)"
    run mkdir -p "$out"
    run $PY -m qlcm.curriculum.build_medical_qa_stage4 \
      --snomed "${SNOMED_SL_TRAIN:-/data/muhsen/repos/finalize-dataset/abdelrahman_rl/final_sl_subsampled.parquet}" \
      --icd    "${ICD_SL_TRAIN:-/home/mbinakba/mcs-instruct-dataset/avey_icd_single_label_rl_train_dataset.parquet}" \
      --output "$out"
    log "stage 4: scoring stage-3 rollouts for bench retention"
    run $PY -m qlcm.curriculum.stats_medical_qa_bench_retention \
      --rollout_dir "$rd" --source "$DATA_DIR/medical_qa_stage3/train.parquet" \
      --tokenizer "$tok" --out "$out/_bench_retention_cache.json"
    log "stage 4: folding retention into train.parquet"
    run $PY -m qlcm.curriculum.build_medical_qa_stage4_retention \
      --existing_train "$out/train.parquet" \
      --medical_cache  "$DATA_DIR/medical_qa_stage3/_stats_cache.json" \
      --bench_cache    "$out/_bench_retention_cache.json" \
      --general_dir    "$DATA_DIR/train" \
      --medical_source "$MEDICAL_QA_SRC/train.parquet" \
      --bench_source   "$DATA_DIR/medical_qa_stage3/train.parquet" \
      --output_dir     "$out" --threshold "$THRESHOLD" --seed 1234
    ;;

  56)
    # Merged stage 5+6. train.parquet and val.parquet are pre-built (the union
    # of both stages' coding pools). Retention follows the STAGE-5 recipe against
    # stage-4 rollouts — stage-6's retention is unavailable by construction,
    # since it is derived from stage-5 rollouts a merged run never produces.
    local out="$DATA_DIR/medical_qa_stage56"
    _require "the merged 5+6 stage" "$out/train.parquet" "$out/val.parquet"
    [[ -f "$out/retention.parquet" ]] && { log "stage 56 retention already built"; return 0; }
    local rd; rd="$(_rollout_dir 4)"
    _have_rollouts "$rd" || { echo "ERROR: no rollout dumps in $rd" >&2; return 1; }
    local tok; tok="$(_tokenizer_for 4)"
    log "stage 56: scoring stage-4 rollouts for the coding carry-forward pool"
    run $PY -m qlcm.curriculum.stats_medical_qa_stage5_retention \
      --rollout_dir "$rd" \
      --coding_source "$DATA_DIR/medical_qa_stage4/train_coding_only.parquet" \
      --bench_source  "$DATA_DIR/medical_qa_stage3/train.parquet" \
      --tokenizer "$tok" --out "$out/_stage4_retention_cache.json"
    log "stage 56: building retention pool"
    run $PY -m qlcm.curriculum.build_medical_qa_stage5_retention \
      --output_dir "$out" \
      --general_dir    "$DATA_DIR/train" \
      --medical_source "$MEDICAL_QA_SRC/train.parquet" \
      --bench_source   "$DATA_DIR/medical_qa_stage3/train.parquet" \
      --coding_source  "$DATA_DIR/medical_qa_stage4/train_coding_only.parquet" \
      --medical_cache  "$DATA_DIR/medical_qa_stage3/_stats_cache.json" \
      --bench_cache    "$DATA_DIR/medical_qa_stage4/_bench_retention_cache.json" \
      --stage4_cache   "$out/_stage4_retention_cache.json" \
      --threshold "$THRESHOLD" --seed 5051
    harmonise "$out/train.parquet" "$out/retention.parquet"
    ;;

  5)
    local out="$DATA_DIR/medical_qa_stage5"
    [[ -f "$out/retention.parquet" ]] && { log "stage 5 data already built"; return 0; }
    local rd; rd="$(_rollout_dir 4)"
    _have_rollouts "$rd" || { echo "ERROR: no rollout dumps in $rd" >&2; return 1; }
    local tok; tok="$(_tokenizer_for 4)"
    log "stage 5: building ICD + SNOMED (multilabel + instruction-following) train/val"
    run mkdir -p "$out"
    run $PY -m qlcm.curriculum.build_medical_qa_stage5 --output "$out"
    log "stage 5: scoring stage-4 rollouts for the coding carry-forward pool"
    run $PY -m qlcm.curriculum.stats_medical_qa_stage5_retention \
      --rollout_dir "$rd" \
      --coding_source "$DATA_DIR/medical_qa_stage4/train_coding_only.parquet" \
      --bench_source  "$DATA_DIR/medical_qa_stage3/train.parquet" \
      --tokenizer "$tok" --out "$out/_stage4_retention_cache.json"
    log "stage 5: building retention pool"
    run $PY -m qlcm.curriculum.build_medical_qa_stage5_retention \
      --output_dir "$out" \
      --general_dir    "$DATA_DIR/train" \
      --medical_source "$MEDICAL_QA_SRC/train.parquet" \
      --bench_source   "$DATA_DIR/medical_qa_stage3/train.parquet" \
      --coding_source  "$DATA_DIR/medical_qa_stage4/train_coding_only.parquet" \
      --medical_cache  "$DATA_DIR/medical_qa_stage3/_stats_cache.json" \
      --bench_cache    "$DATA_DIR/medical_qa_stage4/_bench_retention_cache.json" \
      --stage4_cache   "$out/_stage4_retention_cache.json" \
      --threshold "$THRESHOLD" --seed 5051
    harmonise "$out/train.parquet" "$out/retention.parquet"
    ;;

  6)
    local out="$DATA_DIR/medical_qa_stage6"
    [[ -f "$out/retention.parquet" ]] && { log "stage 6 data already built"; return 0; }
    local rd; rd="$(_rollout_dir 5)"
    _have_rollouts "$rd" || { echo "ERROR: no rollout dumps in $rd" >&2; return 1; }
    local tok; tok="$(_tokenizer_for 5)"
    log "stage 6: building REAL clinical-note coding train/val"
    run mkdir -p "$out"
    run $PY -m qlcm.curriculum.build_medical_qa_stage6 --output "$out"
    log "stage 6: scoring stage-5 rollouts for the unmastered coding pool"
    run $PY -m qlcm.curriculum.stats_medical_qa_stage6_retention \
      --rollout_dir "$rd" \
      --coding_source "$DATA_DIR/medical_qa_stage5/train.parquet" \
      --tokenizer "$tok" --out "$out/_stage5_retention_cache.json"
    log "stage 6: building retention pool"
    run $PY -m qlcm.curriculum.build_medical_qa_stage6_retention \
      --output_dir "$out" \
      --general_dir          "$DATA_DIR/train" \
      --medical_source       "$MEDICAL_QA_SRC/train.parquet" \
      --stage5_coding_source "$DATA_DIR/medical_qa_stage5/train.parquet" \
      --medical_cache        "$DATA_DIR/medical_qa_stage3/_stats_cache.json" \
      --stage5_cache         "$out/_stage5_retention_cache.json" \
      --threshold "$THRESHOLD" --seed 6051
    harmonise "$out/train.parquet" "$out/retention.parquet"
    ;;

  8)
    # Stage 8 branches off stage 6 and trains on coding data ALONE — no
    # retention, so there is nothing to harmonise and no rollout scoring.
    local out="$DATA_DIR/medical_qa_stage8"
    [[ -f "$out/train.parquet" ]] && { log "stage 8 data already built"; return 0; }
    log "stage 8: building coding-only train/val (no retention)"
    run mkdir -p "$out"
    run $PY -m qlcm.curriculum.build_medical_qa_stage8 --output "$out"
    ;;

  esac
}

# verl concat-loads multiple train parquets, so they must share one Arrow schema.
harmonise() {
  log "harmonising train + retention schemas"
  run $PY -m qlcm.curriculum.harmonise_train_files "$@" --drop-conflicts \
      --report "$(dirname "$1")/_harmonise_report.json"
}


# ===========================================================================
# Main loop
# ===========================================================================

_started=0
for STAGE in "${STAGES[@]}"; do
  [[ "$STAGE" == "$START_STAGE" ]] && _started=1
  (( _started )) || continue

  EXP="$(_exp_name "$STAGE")"
  SCRIPT="$(_train_script "$STAGE")"

  echo
  echo "================================================================"
  echo "  QLCM stage: $STAGE   ($EXP)"
  echo "================================================================"

  build_data_for "$STAGE"

  # Where this stage starts from.
  if [[ -n "$INIT_MODEL" && "$STAGE" == "$START_STAGE" ]]; then
    # Explicit pin. Accept either a merged HF dir or a raw global_step_N dir.
    if [[ -f "$INIT_MODEL/config.json" ]]; then
      STAGE_MODEL="$INIT_MODEL"
    elif [[ -d "$INIT_MODEL/actor" ]]; then
      STAGE_MODEL="$(_merge_ckpt "$INIT_MODEL")"
    else
      echo "ERROR: INIT_MODEL=$INIT_MODEL is neither a merged HF dir (config.json)" >&2
      echo "       nor an FSDP checkpoint dir (actor/)." >&2
      exit 1
    fi
    log "stage $STAGE starts from PINNED INIT_MODEL -> $STAGE_MODEL"
    # Only the first stage is pinned; the rest chain normally.
    INIT_MODEL=""
  elif [[ "$STAGE" == "general" ]]; then
    STAGE_MODEL="$BASE_MODEL"
    _require "the general stage" "$STAGE_MODEL/config.json"
  else
    PREV_IDX=0
    for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$STAGE" ]] && PREV_IDX=$(( i - 1 )); done
    PREV="${STAGES[$PREV_IDX]}"
    STAGE_MODEL="$(_stage_model "$PREV")"
    log "stage $STAGE starts from stage $PREV -> $STAGE_MODEL"
  fi

  # Resume inside this stage if it already has checkpoints (crash recovery).
  RESUME_FROM_PATH="${RESUME_FROM_PATH:-$(_latest_ckpt "$CHECKPOINT_ROOT/$EXP")}"
  if [[ -n "$RESUME_FROM_PATH" ]]; then
    log "stage $STAGE resuming from $RESUME_FROM_PATH"
  fi

  MODEL_PATH="$STAGE_MODEL" \
  PROJECT_NAME="$PROJECT_NAME" \
  EXP_NAME="$EXP" \
  DATA_DIR="$DATA_DIR" \
  MEDICAL_QA_SRC="$MEDICAL_QA_SRC" \
  RESUME_FROM_PATH="$RESUME_FROM_PATH" \
  run bash "$SCRIPT" "$@"

  # Clear so the next stage doesn't inherit this stage's resume path.
  unset RESUME_FROM_PATH

  log "stage $STAGE training complete"
  [[ "$SKIP_MERGE" == "1" ]] || _stage_model "$STAGE" > /dev/null

  [[ "$STAGE" == "$END_STAGE" ]] && break
done

echo
FINAL="$(_latest_ckpt "$CHECKPOINT_ROOT/$(_exp_name 8)" 2>/dev/null || true)"
if [[ -n "$FINAL" ]]; then
  echo "pipeline done — final model: $FINAL/merged_hf_model"
else
  echo "pipeline done (stages $START_STAGE..$END_STAGE)"
fi
