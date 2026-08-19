#!/usr/bin/env bash
# Fully-async GRPO on a SUBSAMPLED chat + ifeval + math mix, SINGLE NODE.
#
# Based on full_mix/train_fully_async_no_math.sh, re-scoped for one 8-GPU box:
#   - 4 GPUs rollout / 4 GPUs trainer (disjoint pools on the SAME node — the
#     layout used by verl's own dapo_7b_math_fsdp2_4_4.sh reference script)
#   - actor = the chat-math SFT'd Qwen3-4B in /data/abdelrahman/qwen-sft/...
#   - data  = data/chat_ifeval_math_mix, chat + ifeval ONLY (math excluded from
#     training). Produced by full_mix/curriculum/build_chat_ifeval_math_mix.py
#   - GEN_TP=1 (4 independent vLLM engines) and SP_SIZE=1 — a 4B model needs
#     neither tensor- nor sequence-parallelism on H100s, and 4 engines beat one
#     TP=4 engine on rollout throughput.
#   - MAX_RESPONSE_LEN=8192 (vs 16384 upstream): this mix is chat/IF/short-CoT
#     math from a non-thinking SFT model, so the extra 8k only bought straggler
#     tail. Raise it if response_length/mean starts pinning at the cap.
#
# Docs: https://verl.readthedocs.io/en/latest/advance/fully_async.html
# Run INSIDE the container started by dev/dev.sh:
#   bash dev/dev.sh
#   export FULL_MIX_JUDGE_API_KEY=...      # chat reward is LLM-as-judge
#   export WANDB_API_KEY=...               # optional; console-only without it
#   bash full_mix/train_fully_async_chat_ifeval_math.sh
#
# NOTE on `ray job submit`: full_mix/runtime_env.yaml pins MODEL_PATH (and
# CHAT_VARIANT) for the medical-QA runs. If you launch this script through it,
# those values WIN over the defaults below — pass MODEL_PATH explicitly.

set -xeuo pipefail

# vLLM async rollout engine requires the v1 path.
export VLLM_USE_V1=1

# Load secrets (gitignored). Copy full_mix/secrets.env.example → secrets.env.
_FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_FM_DIR/secrets.env" ]]; then
  source "$_FM_DIR/secrets.env"
fi


# ---- model ----
# The chat-math SFT run under /data/abdelrahman/qwen-sft/full_scale/experiment
# ships checkpoint-324 / -648 / -972 (one per epoch) and final/ (== epoch 3).
# We start from `final`; point MODEL_PATH at a checkpoint-N dir to start earlier.
MODEL_PATH=${MODEL_PATH:-/data/abdelrahman/qwen-sft/full_scale/experiment/final}


# ---- data ----
# Subsampled mix. Rebuild / re-sample with:
#   python -m full_mix.curriculum.build_chat_ifeval_math_mix \
#       --n_chat 2000 --n_ifeval 4000 --n_math 2000 --seed 42
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/chat_ifeval_math_mix}
FULL_MIX_DATA_DIR=${FULL_MIX_DATA_DIR:-/data/abdelrahman/verl/data/full_mix}

# Which chat parquet the mix was built from. `chat_with_baseline` is required
# by the pairwise chat reward (it reads extra_info.baseline_response).
CHAT_VARIANT=${CHAT_VARIANT:-chat_with_baseline}
case "$CHAT_VARIANT" in
  chat)               _CHAT_FILE=chat_train.parquet ;;
  chat_with_baseline) _CHAT_FILE=chat_with_baseline_train.parquet ;;
  *) echo "CHAT_VARIANT must be 'chat' or 'chat_with_baseline'; got '$CHAT_VARIANT'" >&2; exit 2 ;;
esac

# math_train.parquet is deliberately NOT trained on (it still exists in
# DATA_DIR). Add it back to this list to restore the 3-way mix.
TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/${_CHAT_FILE},${DATA_DIR}/ifeval_train.parquet]"}
# Held-out evals covering both rule-based domains in the mix. No chat eval set
# exists (chat is judged pairwise against a baseline, not against a reference).
VAL_FILES=${VAL_FILES:-"[${FULL_MIX_DATA_DIR}/eval/gsm8k_eval.parquet,${FULL_MIX_DATA_DIR}/eval/math500_eval.parquet,${FULL_MIX_DATA_DIR}/eval/ifeval_eval.parquet]"}


# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
EXP_NAME=${EXP_NAME:-chat-ifeval-math-mix-4b}


# ---- sequence lengths ----
# Prompt p99 across the mix is ~1.2k tokens; 2048 keeps all but 4 ifeval rows
# (dropped by data.filter_overlong_prompts).
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}


# ---- sampling ----
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-0.7}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}


# ---- GRPO / KL knobs ----
# KL fully off (as in train_fully_async_no_math.sh). Turning either of these on
# instantiates a reference policy on the 4 trainer GPUs — budget for it.
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
KL_COEF=${KL_COEF:-0.0}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}


# ---- PPO clip / loss / optim ----
CLIP_LOW=${CLIP_LOW:-0.2}
CLIP_HIGH=${CLIP_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
LR=${LR:-1e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-20}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
GRAD_CLIP=${GRAD_CLIP:-1.0}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}


# ---- overlong buffer (DAPO-style soft penalty near the response-length cap) ----
# Penalty zone is the LAST OVERLONG_BUFFER_LEN tokens of MAX_RESPONSE_LEN
# (inside the cap, not on top of it) -> [7168, 8192] with these defaults.
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-True}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}


# ---- importance sampling (off-policy correction) ----
USE_ROLLOUT_LOG_PROBS=${USE_ROLLOUT_LOG_PROBS:-True}


# ---- async-training knobs ----
# Per parameter sync the trainer consumes
#   TRIGGER_SYNC_STEP * REQUIRE_BATCHES * TRAIN_MINI_BSZ = 2*2*16 = 64 prompts
#   (= 1024 trajectories at N=16), in TRIGGER_SYNC_STEP*REQUIRE_BATCHES = 4
#   optimizer steps.
# 5996 prompts / 64 => ~93 parameter versions per epoch.
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.25}
TRIGGER_SYNC_STEP=${TRIGGER_SYNC_STEP:-2}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}


# ---- batching ----
# Streaming async: data.train_batch_size is ignored (0) and gen_batch_size is
# fixed to 1. TRAIN_MINI_BSZ counts PROMPTS; the actor sees
# TRAIN_MINI_BSZ * N_SAMPLES_PER_PROMPT = 256 trajectories per optimizer step.
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-0}
GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-1}
TRAIN_MINI_BSZ=${TRAIN_MINI_BSZ:-16}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-100000}   # epochs are the real bound
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
# TEST_FREQ / SAVE_FREQ count PARAMETER VERSIONS, not optimizer steps.
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-20}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-5}

# Resume policy:
#   RESUME_FROM_PATH non-empty -> RESUME_MODE=resume_path, pass the path through.
#   RESUME_FROM_PATH empty     -> RESUME_MODE=disable (start from MODEL_PATH).
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
if [[ -n "$RESUME_FROM_PATH" ]]; then
  RESUME_MODE=${RESUME_MODE:-resume_path}
else
  RESUME_MODE=${RESUME_MODE:-disable}
fi


# ---- rollout / validation dump dirs ----
# Under /data (bind-mounted by dev/dev.sh). $HOME inside the container is /root,
# which is NOT mounted — dumps written there die with the container.
DUMP_ROOT=${DUMP_ROOT:-/data/abdelrahman/verl/dumps/chat_ifeval_math_mix}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${DUMP_ROOT}/rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${DUMP_ROOT}/val}


# ---- cluster layout (single node: rollout and trainer on DISJOINT GPUs) ----
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-4}
NGPUS_TRAIN=${NGPUS_TRAIN:-4}


# ---- parallelism (dynamic-bsz + vLLM TP) ----
# 4B params: FSDP2 over the 4 trainer GPUs is enough; no Ulysses SP, no vLLM TP.
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 1 ))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
GEN_TP=${GEN_TP:-1}
SP_SIZE=${SP_SIZE:-2}
FSDP_SIZE=${FSDP_SIZE:-${NGPUS_TRAIN}}


# ---- offloading ----
# A 4B model's sharded params+optimizer are ~16 GB/GPU across 4 H100s, so
# offload only costs wall-clock here. Flip ACTOR_OFFLOAD=True if you raise
# MAX_RESPONSE_LEN / ACTOR_PPO_MAX_TOKEN_LEN far enough to OOM.
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-False}
REF_OFFLOAD=${REF_OFFLOAD:-True}


# ---- vLLM rollout knobs ----
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}


# ---- reward dispatcher ----
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/full_mix/rewards/compute_score.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}
# Reward managers run as Ray actors; each fans its chunk out over an executor
# thread pool, so judge latency overlaps. Judge-bound mixes want more than the
# verl default of 8.
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}


# ---- judge (the chat slice is scored by an LLM judge) ----
# Defaults mirror full_mix/runtime_env.yaml (OpenRouter). The API KEY is NOT
# defaulted here — export it or put it in secrets.env.
export FULL_MIX_JUDGE_API_BASE=${FULL_MIX_JUDGE_API_BASE:-https://openrouter.ai/api/v1}
export FULL_MIX_JUDGE_API_KEY=${FULL_MIX_JUDGE_API_KEY:-}
export FULL_MIX_JUDGE_MODEL=${FULL_MIX_JUDGE_MODEL:-deepseek/deepseek-v3.2}
export FULL_MIX_JUDGE_TIMEOUT=${FULL_MIX_JUDGE_TIMEOUT:-240}
export FULL_MIX_JUDGE_MAX_RETRIES=${FULL_MIX_JUDGE_MAX_RETRIES:-3}
export FULL_MIX_JUDGE_MAX_TOKENS=${FULL_MIX_JUDGE_MAX_TOKENS:-32768}
export FULL_MIX_JUDGE_FORCE_JSON=${FULL_MIX_JUDGE_FORCE_JSON:-1}
export FULL_MIX_JUDGE_DISABLE_THINKING=${FULL_MIX_JUDGE_DISABLE_THINKING:-0}
export FULL_MIX_JUDGE_CONCURRENCY=${FULL_MIX_JUDGE_CONCURRENCY:-32}

# Global identity gate: an extra judge call on EVERY non-identity sample that
# scored > 0, zeroing the reward if the response claims a non-Avey identity.
# OFF here — this mix carries no identity data and the SFT actor was never
# identity-trained, so the gate would only add judge cost and reward noise.
# Set to 1 to restore the full_mix default.
export FULL_MIX_GLOBAL_IDENTITY_CHECK_ENABLED=${FULL_MIX_GLOBAL_IDENTITY_CHECK_ENABLED:-0}


# ---- preflight ----
# Fail fast instead of burning GPU-hours on degenerate rewards.
set +x
_die() { echo "[preflight] $*" >&2; exit 2; }

[[ -f "${MODEL_PATH}/config.json" ]] || _die "MODEL_PATH has no config.json: ${MODEL_PATH}"
for _f in "${DATA_DIR}/${_CHAT_FILE}" "${DATA_DIR}/ifeval_train.parquet" "${DATA_DIR}/math_train.parquet"; do
  [[ -f "$_f" ]] || _die "missing train parquet: $_f
  build it with: python -m full_mix.curriculum.build_chat_ifeval_math_mix"
done

# The pairwise chat reward returns a constant 0.5 when the judge is
# unreachable — no advantage signal, silently. Refuse to start without a key.
if [[ -z "${FULL_MIX_JUDGE_API_KEY}" ]]; then
  _die "FULL_MIX_JUDGE_API_KEY is empty; the chat slice would score a constant 0.5.
  export it (or set it in full_mix/secrets.env) before launching."
fi

LOGGER=${LOGGER:-'["console","wandb"]'}
if [[ -z "${WANDB_API_KEY:-}" && "${LOGGER}" == *wandb* ]]; then
  echo "[preflight] WANDB_API_KEY unset -> logging to console only." >&2
  LOGGER='["console"]'
fi
set -x


python3 -m verl.experimental.fully_async_policy.fully_async_main \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD}" \
  algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
  algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.prompt_key=prompt \
  data.truncation='error' \
  data.filter_overlong_prompts=True \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  data.gen_batch_size="${GEN_PROMPT_BSZ}" \
  data.return_raw_chat=True \
  \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}" \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_MINI_BSZ}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.actor.use_rollout_log_probs="${USE_ROLLOUT_LOG_PROBS}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OFFLOAD}" \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
  actor_rollout_ref.rollout.temperature="${TEMP}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${TEMP}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_OFFLOAD}" \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
  \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  reward.reward_manager.name=dapo \
  reward.custom_reward_function.path="${REWARD_FN_PATH}" \
  reward.custom_reward_function.name="${REWARD_FN_NAME}" \
  +reward.reward_kwargs.overlong_buffer_cfg.enable="${ENABLE_OVERLONG_BUFFER}" \
  +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}" \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR}" \
  +reward.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LEN}" \
  \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.val_before_train=True \
  trainer.resume_mode="${RESUME_MODE}" \
  ${RESUME_FROM_PATH:++trainer.resume_from_path=${RESUME_FROM_PATH}} \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.nnodes="${NNODES_TRAIN}" \
  trainer.n_gpus_per_node="${NGPUS_TRAIN}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.test_freq="${TEST_FREQ}" \
  \
  rollout.nnodes="${NNODES_ROLLOUT}" \
  rollout.n_gpus_per_node="${NGPUS_ROLLOUT}" \
  rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
  \
  async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
  async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}" \
  async_training.require_batches="${REQUIRE_BATCHES}" \
  async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
  ${RAY_ENV_FLAGS:-} \
  "$@"
