#!/usr/bin/env bash
# Fully-async GRPO training on Medical QA + Medical Conversation data.
#
# Self-contained experiment — NOT the same as train_fully_async_no_math.sh.
# Differences from no_math:
#   - Dataset:        data/medical_qa/{train,val}.parquet
#                       (medical_qa + medical_conv rows, schema produced
#                        by hazem's medical-data-gen pipeline)
#   - Reward fn:      full_mix/rewards/qa_openrouter.py — OpenRouter judge
#                       (OpenAI-compatible HTTP) over QA key points and
#                       conversation rubric. Prompts + scoring identical to
#                       qa_bedrock; only the transport changed.
#   - KL regime:      KL-IN-REWARD with a small coef (was off everywhere
#                       else). use_kl_loss stays False.
#   - Judge env:      QA_JUDGE_OPENROUTER_API_KEY (or OPENROUTER_API_KEY)
#                       + QA_JUDGE_MODEL (OpenRouter "<provider>/<model>"
#                       id). Retention judges use FULL_MIX_JUDGE_* —
#                       qa_openrouter doesn't read them.
#
# Everything else (cluster layout, sampling, batching, overlong buffer,
# PPO clip / loss, vLLM, FSDP, offload) is inherited from the same defaults
# as train_fully_async_no_math.sh.
#
# Run INSIDE the container started by dev/dev.sh.

set -xeuo pipefail

export VLLM_USE_V1=1

_FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_FM_DIR/secrets.env" ]]; then
  source "$_FM_DIR/secrets.env"
fi


# ---- model ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}


# ---- data ----
# Training mix: medical_qa primary task + curated full_mix RETENTION samples
# (hard + sampled-easy prompts from a previous run, intended to keep the model
# from regressing on its previously-trained tasks during medical fine-tuning).
# Disable retention by overriding TRAIN_FILES to just the medical parquet.
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data}
CURATED_DIR=${CURATED_DIR:-${DATA_DIR}/full_mix/curated}
TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/medical_qa/train.parquet,${CURATED_DIR}/ifeval_train.parquet,${CURATED_DIR}/chat_with_baseline_train.parquet,${CURATED_DIR}/safety_train.parquet,${CURATED_DIR}/identity_train.parquet]"}
VAL_FILES=${VAL_FILES:-"[${DATA_DIR}/medical_qa/val.parquet]"}


# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
EXP_NAME=${EXP_NAME:-medical-qa-fresh}


# ---- sequence lengths ----
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}


# ---- sampling ----
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-0.7}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}


# ---- GRPO / KL knobs ----
# Medical QA runs with KL-IN-REWARD (token-level KL penalty added into the
# reward signal) at a small coefficient, NOT KL-in-loss. This is the
# user-specified regime for this stage.
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-True}
KL_COEF=${KL_COEF:-0.0001}
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
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-True}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}


# ---- importance sampling (off-policy correction) ----
USE_ROLLOUT_LOG_PROBS=${USE_ROLLOUT_LOG_PROBS:-True}


# ---- async-training knobs ----
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.25}
TRIGGER_SYNC_STEP=${TRIGGER_SYNC_STEP:-4}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}


# ---- batching ----
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-0}
GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-1}
TRAIN_MINI_BSZ=${TRAIN_MINI_BSZ:-16}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-1000000}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-20}

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
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$HOME/verl_dumps/medical_qa_rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$HOME/verl_dumps/medical_qa_val}


# ---- cluster layout (rollout and trainer on DISJOINT nodes) ----
NNODES_ROLLOUT=${NNODES_ROLLOUT:-3}
NNODES_TRAIN=${NNODES_TRAIN:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# ---- parallelism (dynamic-bsz + Ulysses SP + vLLM TP) ----
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 1 ))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
GEN_TP=${GEN_TP:-4}
SP_SIZE=${SP_SIZE:-4}
FSDP_SIZE=${FSDP_SIZE:-16}


# ---- offloading ----
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-True}
REF_OFFLOAD=${REF_OFFLOAD:-True}


# ---- vLLM rollout knobs ----
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}


# ---- reward dispatcher ----
# Mixed dispatcher: medical_qa / medical_conv rows -> qa_openrouter,
# curated retention rows (ifeval/chat/safety/identity) -> the full_mix
# dispatcher (which still applies the global identity gate to non-identity
# samples). To train on medical only with no retention mix, override:
#   REWARD_FN_PATH=/workspace/verl/full_mix/rewards/qa_openrouter.py
#   TRAIN_FILES="[${DATA_DIR}/medical_qa/train.parquet]"
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/full_mix/rewards/compute_score_medical_mix.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}


# ---- judges ----
# Two independent judge endpoints:
#   * Medical samples (qa_openrouter) -> OpenRouter.             Reads QA_JUDGE_*
#   * Retention samples (full_mix compute_score: chat/safety/identity) ->
#     OpenRouter / vLLM-served judge.                            Reads FULL_MIX_JUDGE_*
# Required keys must come from secrets.env (gitignored).

# OpenRouter (qa_openrouter). Resolution order inside the module:
#   1. QA_JUDGE_OPENROUTER_API_KEY (judge-specific override)
#   2. OPENROUTER_API_KEY          (reuses an existing OpenRouter key)
# Either var being set in secrets.env is sufficient.
export QA_JUDGE_OPENROUTER_API_KEY=${QA_JUDGE_OPENROUTER_API_KEY:-${OPENROUTER_API_KEY:-}}
# Optional: only set this if you're routing through a proxy or self-hosting.
# Default inside qa_openrouter.py is https://openrouter.ai/api/v1.
# export QA_JUDGE_OPENROUTER_BASE_URL=...
export QA_JUDGE_MODEL=${QA_JUDGE_MODEL:-openai/gpt-5.4-mini}
export QA_JUDGE_MAX_TOKENS=${QA_JUDGE_MAX_TOKENS:-32768}
# To switch to direct OpenAI: revert the dispatcher's import in
# full_mix/rewards/compute_score_medical_mix.py to qa_openai, and export
# QA_JUDGE_OPENAI_API_KEY (or OPENAI_API_KEY) + QA_JUDGE_MODEL=gpt-5.4-mini.
# To switch back to qa_bedrock: revert the dispatcher import to qa_bedrock,
# and export QA_JUDGE_AWS_ACCESS_KEY / _SECRET_KEY / _REGION instead.

# OpenRouter / vLLM (chat / safety / identity / global identity gate)
export FULL_MIX_JUDGE_API_BASE=${FULL_MIX_JUDGE_API_BASE:-http://127.0.0.1:8000/v1}
export FULL_MIX_JUDGE_API_KEY=${FULL_MIX_JUDGE_API_KEY:-EMPTY}
export FULL_MIX_JUDGE_MODEL=${FULL_MIX_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export FULL_MIX_JUDGE_TIMEOUT=${FULL_MIX_JUDGE_TIMEOUT:-120}
export FULL_MIX_JUDGE_MAX_RETRIES=${FULL_MIX_JUDGE_MAX_RETRIES:-10}
export FULL_MIX_JUDGE_MAX_TOKENS=${FULL_MIX_JUDGE_MAX_TOKENS:-32768}
export FULL_MIX_JUDGE_FORCE_JSON=${FULL_MIX_JUDGE_FORCE_JSON:-1}
export FULL_MIX_JUDGE_DISABLE_THINKING=${FULL_MIX_JUDGE_DISABLE_THINKING:-0}
export FULL_MIX_JUDGE_CONCURRENCY=${FULL_MIX_JUDGE_CONCURRENCY:-32}
export FULL_MIX_IDENTITY_JUDGE_MODEL=${FULL_MIX_IDENTITY_JUDGE_MODEL:-}


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
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.val_before_train=True \
  trainer.resume_mode="${RESUME_MODE}" \
  ${RESUME_FROM_PATH:++trainer.resume_from_path=${RESUME_FROM_PATH}} \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.nnodes="${NNODES_TRAIN}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.test_freq="${TEST_FREQ}" \
  \
  rollout.nnodes="${NNODES_ROLLOUT}" \
  rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
  rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
  \
  async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
  async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}" \
  async_training.require_batches="${REQUIRE_BATCHES}" \
  async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
  ${RAY_ENV_FLAGS:-} \
  "$@"
