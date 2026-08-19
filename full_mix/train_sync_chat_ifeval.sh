#!/usr/bin/env bash
# SYNCHRONOUS (colocate) GRPO on the chat + ifeval mix, SINGLE NODE.
#
# Sibling of full_mix/train_fully_async_chat_ifeval_math.sh. Same model, same
# data, same reward dispatcher — the difference is the execution model:
#
#   fully-async : rollout and trainer own DISJOINT GPUs (4+4) and overlap.
#   this script : all 8 GPUs are shared. vLLM generates, sleeps, the actor
#                 trains, weights reload, repeat. No overlap, no staleness,
#                 every batch is on-policy.
#
# Keep the knobs below in sync with the async script when comparing the two;
# the batch arithmetic is deliberately matched (64 prompts / 4 optimizer steps
# between weight updates).
#
# Run INSIDE the container started by dev/dev.sh:
#   bash dev/dev.sh
#   export FULL_MIX_JUDGE_API_KEY=...      # chat reward is LLM-as-judge
#   bash full_mix/train_sync_chat_ifeval.sh

set -xeuo pipefail

export VLLM_USE_V1=1

_FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_FM_DIR/secrets.env" ]]; then
  source "$_FM_DIR/secrets.env"
fi

export WANDB_API_KEY=${WANDB_API_KEY:-}

# ---- model ----
MODEL_PATH=${MODEL_PATH:-/data/abdelrahman/qwen-sft/full_scale/experiment/final}


# ---- data ----
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/chat_ifeval_math_mix}
FULL_MIX_DATA_DIR=${FULL_MIX_DATA_DIR:-/data/abdelrahman/verl/data/full_mix}

CHAT_VARIANT=${CHAT_VARIANT:-chat_with_baseline}
case "$CHAT_VARIANT" in
  chat)               _CHAT_FILE=chat_train.parquet ;;
  chat_with_baseline) _CHAT_FILE=chat_with_baseline_train.parquet ;;
  *) echo "CHAT_VARIANT must be 'chat' or 'chat_with_baseline'; got '$CHAT_VARIANT'" >&2; exit 2 ;;
esac

# math_train.parquet is deliberately NOT trained on (it still exists in
# DATA_DIR). 2000 chat + 3996 ifeval = 5996 prompts after overlong filtering.
TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/${_CHAT_FILE},${DATA_DIR}/ifeval_train.parquet]"}
# Math evals kept for retention tracking even though math is not trained.
VAL_FILES=${VAL_FILES:-"[${FULL_MIX_DATA_DIR}/eval/gsm8k_eval.parquet,${FULL_MIX_DATA_DIR}/eval/math500_eval.parquet,${FULL_MIX_DATA_DIR}/eval/ifeval_eval.parquet]"}


# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
EXP_NAME=${EXP_NAME:-chat-ifeval-sync-4b}


# ---- sequence lengths ----
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}


# ---- sampling ----
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-0.7}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}


# ---- GRPO / KL knobs ----
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


# ---- overlong buffer ----
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-True}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}


# ---- batching (SYNC semantics — differs from the async script) ----
# TRAIN_BATCH_SIZE prompts are generated, scored, then trained on before the
# next generation. TRAIN_MINI_BSZ splits that into optimizer steps:
#   64 / 16 = 4 optimizer steps per generation, matching the async script's
#   4 steps per parameter sync.
# 5996 prompts / 64 => ~93 steps per epoch.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
TRAIN_MINI_BSZ=${TRAIN_MINI_BSZ:-16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
# In the sync trainer TEST_FREQ / SAVE_FREQ count TRAINING STEPS.
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-20}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-5}

RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
if [[ -n "$RESUME_FROM_PATH" ]]; then
  RESUME_MODE=${RESUME_MODE:-resume_path}
else
  RESUME_MODE=${RESUME_MODE:-disable}
fi


# ---- rollout / validation dump dirs ----
DUMP_ROOT=${DUMP_ROOT:-/data/abdelrahman/verl/dumps/chat_ifeval_sync}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${DUMP_ROOT}/rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${DUMP_ROOT}/val}


# ---- cluster layout (colocate: ALL GPUs run both rollout and training) ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# ---- parallelism ----
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 1 ))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
GEN_TP=${GEN_TP:-1}
SP_SIZE=${SP_SIZE:-2}
FSDP_SIZE=${FSDP_SIZE:-${NGPUS_PER_NODE}}


# ---- offloading (REQUIRED in colocate) ----
# FSDP state and the vLLM engine share every GPU, so actor params + optimizer
# go to CPU while vLLM generates. Leaving these False is what makes colocate
# OOM. free_cache_engine additionally drops the KV cache during training.
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-True}
REF_OFFLOAD=${REF_OFFLOAD:-True}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}


# ---- vLLM rollout knobs ----
# Lower than the async script's 0.90: there the rollout GPUs were vLLM-only.
# Here the same GPU also has to hold gathered FSDP params and gradients.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}


# ---- reward dispatcher ----
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/full_mix/rewards/compute_score.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}


# ---- judge (the chat slice is scored by an LLM judge) ----
export FULL_MIX_JUDGE_API_BASE=${FULL_MIX_JUDGE_API_BASE:-https://openrouter.ai/api/v1}
export FULL_MIX_JUDGE_API_KEY=${FULL_MIX_JUDGE_API_KEY:-}
export FULL_MIX_JUDGE_MODEL=${FULL_MIX_JUDGE_MODEL:-deepseek/deepseek-v3.2}
export FULL_MIX_JUDGE_TIMEOUT=${FULL_MIX_JUDGE_TIMEOUT:-240}
export FULL_MIX_JUDGE_MAX_RETRIES=${FULL_MIX_JUDGE_MAX_RETRIES:-3}
export FULL_MIX_JUDGE_MAX_TOKENS=${FULL_MIX_JUDGE_MAX_TOKENS:-32768}
export FULL_MIX_JUDGE_FORCE_JSON=${FULL_MIX_JUDGE_FORCE_JSON:-1}
export FULL_MIX_JUDGE_DISABLE_THINKING=${FULL_MIX_JUDGE_DISABLE_THINKING:-0}
export FULL_MIX_JUDGE_CONCURRENCY=${FULL_MIX_JUDGE_CONCURRENCY:-32}

if [[ -z "${FULL_MIX_JUDGE_API_KEY}" ]]; then
  echo "[preflight] FULL_MIX_JUDGE_API_KEY is empty; the chat slice would score a constant 0.5." >&2
  echo "[preflight] Set it in full_mix/secrets.env (see secrets.env.example) before launching." >&2
  exit 2
fi

# Global identity gate off — this mix has no identity data.
export FULL_MIX_GLOBAL_IDENTITY_CHECK_ENABLED=${FULL_MIX_GLOBAL_IDENTITY_CHECK_ENABLED:-0}


python3 -m verl.trainer.main_ppo \
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
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  \
  actor_rollout_ref.hybrid_engine=True \
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
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OFFLOAD}" \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
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
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.val_before_train=True \
  trainer.resume_mode="${RESUME_MODE}" \
  ${RESUME_FROM_PATH:++trainer.resume_from_path=${RESUME_FROM_PATH}} \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.test_freq="${TEST_FREQ}" \
  ${RAY_ENV_FLAGS:-} \
  "$@"
