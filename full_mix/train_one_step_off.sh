#!/usr/bin/env bash
# One-step-off-policy GRPO for the full_mix harness.
#
# Why: with train.sh (standard PPO), rollout and training run SEQUENTIALLY on
# the same GPUs — gen takes ~11 min / step, update_actor ~3 min, and nothing
# happens on the rollout GPUs while the actor is updating. One-step-off splits
# the GPUs into a *rollout pool* and a *training pool* and pipelines them:
# while step N's actor update is running, rollout for step N+1 is already
# generating against the old weights. Weights are synced at the end of each
# training step via a Ray collective group. The "off-by-one" policy lag is
# normally benign for GRPO.
#
# Docs: https://verl.readthedocs.io/en/latest/advance/one_step_off.html
# Reference script: verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_4_12.sh
#
# Run this INSIDE the container started by dev/dev.sh.

set -x

# Load secrets (gitignored). Copy full_mix/secrets.env.example → secrets.env.
_FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_FM_DIR/secrets.env" ]]; then
  source "$_FM_DIR/secrets.env"
fi

# ---- model ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}

# ---- data ----
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/full_mix}
TRAIN_FILES="[${DATA_DIR}/train/math_train.parquet,${DATA_DIR}/train/ifeval_train.parquet,${DATA_DIR}/train/chat_train.parquet,${DATA_DIR}/train/safety_train.parquet]"
# IFBench dropped — constraint IDs not in full_mix/ifeval/FUNCTION_DICT.
VAL_FILES="[${DATA_DIR}/eval/gsm8k_eval.parquet,${DATA_DIR}/eval/math500_eval.parquet,${DATA_DIR}/eval/ifeval_eval.parquet]"

# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
EXP_NAME=${EXP_NAME:-full-mix-grpo-one-step-off}

# ---- sequence lengths ----
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}

# ---- sampling ----
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-0.7}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}

# ---- batching ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-192}

# ---- GRPO knobs ----
CLIP_LOW=${CLIP_LOW:-0.2}
CLIP_HIGH=${CLIP_HIGH:-0.28}
FILTER_METRIC=${FILTER_METRIC:-seq_reward}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-10}

# ---- resource split (default: 8-GPU node carved 6 rollout + 2 training) ----
#   N_GPUS_ROLLOUT : GPUs dedicated to vLLM generation workers
#   N_GPUS_TRAINING: GPUs dedicated to FSDP actor training
# The user wanted MORE gen resources than training; tune via env vars.
NNODES=${NNODES:-1}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-6}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-2}

# ---- rollout engine parallelism ----
# For a 4B model, TP=1 maximizes vLLM instance count and avoids TP communication.
ROLLOUT_TP=${ROLLOUT_TP:-1}

# ---- actor FSDP/SP sizes (must divide N_GPUS_TRAINING) ----
FSDP_SIZE=${FSDP_SIZE:-${N_GPUS_TRAINING}}
SP_SIZE=${SP_SIZE:-1}

# ---- offloading (enable when training pool is small to fit 4B + adam + acts) ----
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-True}
REF_OFFLOAD=${REF_OFFLOAD:-True}

# ---- reward dispatcher ----
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/full_mix/rewards/compute_score.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}

# ---- judge (OpenRouter by default; swap to self-hosted vLLM via env) ----
export FULL_MIX_JUDGE_API_BASE=${FULL_MIX_JUDGE_API_BASE:-https://openrouter.ai/api/v1}
export FULL_MIX_JUDGE_API_KEY=${FULL_MIX_JUDGE_API_KEY:-}
export FULL_MIX_JUDGE_MODEL=${FULL_MIX_JUDGE_MODEL:-x-ai/grok-4.1-fast}
export FULL_MIX_JUDGE_TIMEOUT=${FULL_MIX_JUDGE_TIMEOUT:-120}
export FULL_MIX_JUDGE_MAX_RETRIES=${FULL_MIX_JUDGE_MAX_RETRIES:-3}
export FULL_MIX_JUDGE_MAX_TOKENS=${FULL_MIX_JUDGE_MAX_TOKENS:-4096}
export FULL_MIX_JUDGE_FORCE_JSON=${FULL_MIX_JUDGE_FORCE_JSON:-1}
# DISABLE_THINKING uses a vLLM-only extra_body field. Keep off for OpenRouter.
export FULL_MIX_JUDGE_DISABLE_THINKING=${FULL_MIX_JUDGE_DISABLE_THINKING:-0}
export FULL_MIX_JUDGE_CONCURRENCY=${FULL_MIX_JUDGE_CONCURRENCY:-16}

python3 -m verl.experimental.one_step_off_policy.main_ppo \
  algorithm.adv_estimator=grpo \
  \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  +data.gen_batch_size="${GEN_BATCH_SIZE}" \
  \
  +algorithm.filter_groups.enable=True \
  +algorithm.filter_groups.metric="${FILTER_METRIC}" \
  +algorithm.filter_groups.max_num_gen_batches="${MAX_NUM_GEN_BATCHES}" \
  \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode="token-mean" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OFFLOAD}" \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.temperature="${TEMP}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature="${TEMP}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_OFFLOAD}" \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}" \
  \
  algorithm.rollout_correction.bypass_mode=True \
  \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="${REWARD_FN_PATH}" \
  reward.custom_reward_function.name="${REWARD_FN_NAME}" \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
  rollout.nnodes="${NNODES}" \
  rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.total_epochs=1 \
  trainer.rollout_data_dir=$HOME/verl_dumps/full_mix_rollouts_osop \
  trainer.validation_data_dir=$HOME/verl_dumps/full_mix_val_osop \
  ${RAY_ENV_FLAGS} \
  "$@"
