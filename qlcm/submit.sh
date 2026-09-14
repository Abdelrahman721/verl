#!/usr/bin/env bash
# Submit the QLCM pipeline to an already-running multi-node Ray cluster via
# `ray job submit`, with runtime_env.yaml as the environment.
#
# Run this on the HEAD node, inside the dev container. It does not start Ray —
# see multinode.md step 2 for that.
#
#   bash qlcm/submit.sh                          # 2-node default (see below)
#   START_STAGE=general END_STAGE=general bash qlcm/submit.sh
#   FOLLOW=1 bash qlcm/submit.sh                 # stream logs instead of detaching
#
# Anything in "$@" is forwarded to run_pipeline.sh, and from there to the
# training scripts as extra Hydra overrides.
#
# Why a script and not a one-liner: the entrypoint has to cd into the repo,
# carry the topology as env, and quote correctly through `bash -c`. Getting
# that wrong silently launches with the single-node defaults.

set -euo pipefail

_QLCM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(dirname "$_QLCM_DIR")"

# Path INSIDE the container — identical on every node because dev/dev.sh mounts
# the repo at the same place and /data is shared storage.
WORKDIR=${WORKDIR:-/workspace/verl}

RAY_DASHBOARD=${RAY_DASHBOARD:-http://127.0.0.1:8265}
RUNTIME_ENV=${RUNTIME_ENV:-qlcm/runtime_env.yaml}

# ---- topology: 2 x (2 train + 6 rollout) = all 16 GPUs, 12 rollout engines ---
# Symmetric: every machine gives 2 GPUs to the trainer pool and 6 to rollout.
# Rollout is the bottleneck for a 1.7B actor, hence the 4/12 split rather than
# an even one. The two pools stay disjoint. FSDP_SIZE derives from
# NNODES_TRAIN*NGPUS_TRAIN = 4, so the trainer now shards ACROSS both nodes.
NNODES_TRAIN=${NNODES_TRAIN:-2}
NGPUS_TRAIN=${NGPUS_TRAIN:-2}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-2}
NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-6}

START_STAGE=${START_STAGE:-general}
END_STAGE=${END_STAGE:-8}
# Pin the model START_STAGE begins from (a global_step_N dir or a merged HF
# dir). Without it the pipeline auto-resolves the previous stage's LATEST
# checkpoint, which is wrong when you stopped a stage early.
INIT_MODEL=${INIT_MODEL:-}
FOLLOW=${FOLLOW:-1}

cd "$_REPO_ROOT"

# ---- preflight -------------------------------------------------------------
if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "[submit] $RUNTIME_ENV not found. Copy qlcm/runtime_env.yaml.example and fill it in." >&2
  exit 2
fi
# Only QLCM_JUDGE_API_KEY is needed by every stage (chat / safety / identity).
# The others are stage-scoped, so an unset one is a warning, not a blocker —
# the general stage runs fine without them.
if grep -qE '^\s*QLCM_JUDGE_API_KEY:\s*"?REPLACE_ME' "$RUNTIME_ENV"; then
  echo "[submit] QLCM_JUDGE_API_KEY is still REPLACE_ME in $RUNTIME_ENV." >&2
  echo "[submit] Every stage judges chat/safety/identity rows; set it before launching." >&2
  exit 2
fi
_missing=$(grep -nE '^\s*[A-Z_]+:\s*"?REPLACE_ME' "$RUNTIME_ENV" || true)
if [[ -n "$_missing" ]]; then
  echo "[submit] WARNING — placeholders still in $RUNTIME_ENV:"
  echo "$_missing" | sed 's/^/  /'
  echo "[submit]   QA_JUDGE_OPENROUTER_API_KEY : needed from stage 1 (medical QA judge)"
  echo "[submit]   OPENROUTER_API_KEY          : needed from stage 5 (coding IF judge)"
  echo "[submit]   WANDB_API_KEY               : needed for logging"
  echo "[submit] Fine for the general stage; stages that need them will fail at startup."
  echo
fi

# A bogus QA judge key does NOT fail loudly: qa_openrouter_bench exhausts its
# retries and returns {"accuracy": 1, "clarity": 1}, the FLOOR score. Every
# medical row would then carry the same constant reward, which under GRPO is
# zero advantage — the run burns full generation cost and learns nothing. So
# block rather than warn when the run will actually reach a medical stage.
if [[ "$START_STAGE" != "general" ]] && \
   grep -qE '^\s*QA_JUDGE_OPENROUTER_API_KEY:\s*"?REPLACE_ME' "$RUNTIME_ENV"; then
  echo "[submit] START_STAGE=$START_STAGE needs the medical QA judge, but" >&2
  echo "[submit] QA_JUDGE_OPENROUTER_API_KEY is still REPLACE_ME in $RUNTIME_ENV." >&2
  echo "[submit] An invalid key scores every medical row at the floor instead of" >&2
  echo "[submit] failing — the run would look healthy and train on nothing." >&2
  exit 2
fi

NEED=$(( NNODES_TRAIN * NGPUS_TRAIN + NNODES_ROLLOUT * NGPUS_ROLLOUT ))
HAVE=$(ray status 2>/dev/null | grep -oE '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ GPU' | head -1 | sed -E 's#.*/([0-9]+)\.[0-9]+ GPU#\1#')
if [[ -z "${HAVE:-}" ]]; then
  echo "[submit] could not read GPU count from \`ray status\` — is the cluster up?" >&2
  exit 2
fi
echo "[submit] cluster GPUs: $HAVE   requested: $NEED"
echo "[submit]   train  : ${NNODES_TRAIN} x ${NGPUS_TRAIN}"
echo "[submit]   rollout: ${NNODES_ROLLOUT} x ${NGPUS_ROLLOUT}  -> $(( NNODES_ROLLOUT * NGPUS_ROLLOUT )) vLLM engines at GEN_TP=1"
if (( NEED > HAVE )); then
  echo "[submit] not enough GPUs: the two pools must be disjoint and both must fit." >&2
  exit 2
fi

# ---- entrypoint ------------------------------------------------------------
# The topology travels as env on the entrypoint, NOT in runtime_env.yaml — it
# is a per-launch decision, not configuration. Everything else (keys, judge
# model, provider pool) comes from the YAML, which Ray applies to the driver
# and every actor.
# Forward every run_pipeline knob that is set in THIS shell. Without this a
# knob like MERGE_56 is silently dropped: run_pipeline then builds a STAGES
# array that does not contain the requested START_STAGE, matches nothing, and
# exits 0 having done no work.
_FORWARD=(
  MERGE_56 SKIP_STAGE2 SKIP_MERGE THRESHOLD MAX_CKPT_TO_KEEP SAVE_FREQ
  BENCH_PARQUET CARVE_PARQUETS
  DATA_DIR MEDICAL_QA_SRC DUMPS_DIR CHECKPOINT_ROOT PROJECT_NAME BASE_MODEL
  STAGE2_CONV_KEEP_N
  STAGE3_STAGE1_MASTERED_N STAGE3_STAGE2_UNMASTERED_N STAGE3_STAGE2_MASTERED_N
  STAGE3_BENCH_MCQ_N STAGE3_BENCH_NUMERIC_N STAGE3_BENCH_MEDEC_N STAGE3_BENCH_TEXT_N
)
_FWD=""
for _v in "${_FORWARD[@]}"; do
  if [[ -n "${!_v:-}" ]]; then
    # printf %q so values with spaces (CARVE_PARQUETS) survive the bash -c string.
    _FWD+="${_v}=$(printf '%q' "${!_v}") "
  fi
done

ENTRYPOINT="cd ${WORKDIR} && \
NNODES_TRAIN=${NNODES_TRAIN} NGPUS_TRAIN=${NGPUS_TRAIN} \
NNODES_ROLLOUT=${NNODES_ROLLOUT} NGPUS_ROLLOUT=${NGPUS_ROLLOUT} \
START_STAGE=${START_STAGE} END_STAGE=${END_STAGE} \
${INIT_MODEL:+INIT_MODEL=${INIT_MODEL} }${_FWD}\
bash qlcm/run_pipeline.sh"
if (( $# )); then
  ENTRYPOINT="${ENTRYPOINT} $*"
fi

SUBMIT_ARGS=(ray job submit --address "$RAY_DASHBOARD" --runtime-env "$RUNTIME_ENV")
(( FOLLOW )) || SUBMIT_ARGS+=(--no-wait)
SUBMIT_ARGS+=(-- bash -c "$ENTRYPOINT")

echo "[submit] entrypoint: $ENTRYPOINT"
echo
"${SUBMIT_ARGS[@]}"

if ! (( FOLLOW )); then
  echo
  echo "[submit] detached. Follow with:"
  echo "  ray job logs -f --address $RAY_DASHBOARD <submission-id>"
  echo "  ray job list --address $RAY_DASHBOARD"
  echo "  ray job stop --address $RAY_DASHBOARD <submission-id>"
fi
