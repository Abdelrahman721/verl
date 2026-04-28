#!/usr/bin/env bash
# Per-domain diagnostic: train GRPO on ONLY the ifeval subset.
# See train_math_only.sh for context.

set -x

_FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_FM_DIR/secrets.env" ]]; then
  source "$_FM_DIR/secrets.env"
fi

# ---- model ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}

# ---- data ----
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/full_mix}
TRAIN_FILES="[${DATA_DIR}/train/ifeval_train.parquet]"
VAL_FILES="[${DATA_DIR}/eval/ifeval_eval.parquet]"

# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
EXP_NAME=${EXP_NAME:-full-mix-ifeval-only}

# ---- sequence lengths ----
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}

# ---- sampling ----
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}

# ---- batching ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-192}

# ---- GRPO knobs ----
CLIP_LOW=${CLIP_LOW:-0.2}
CLIP_HIGH=${CLIP_HIGH:-0.28}
ROLLOUT_IS_LEVEL=${ROLLOUT_IS_LEVEL:-token}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
FILTER_METRIC=${FILTER_METRIC:-seq_reward}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-3}

# ---- reward dispatcher ----
REWARD_FN_PATH=${REWARD_FN_PATH:-full_mix/rewards/compute_score.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}

# ---- judge (unused for ifeval-only, but harmless to export) ----
export FULL_MIX_JUDGE_API_BASE=${FULL_MIX_JUDGE_API_BASE:-http://127.0.0.1:8000/v1}
export FULL_MIX_JUDGE_API_KEY=${FULL_MIX_JUDGE_API_KEY:-EMPTY}
export FULL_MIX_JUDGE_MODEL=${FULL_MIX_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export FULL_MIX_JUDGE_TIMEOUT=${FULL_MIX_JUDGE_TIMEOUT:-120}
export FULL_MIX_JUDGE_MAX_RETRIES=${FULL_MIX_JUDGE_MAX_RETRIES:-10}
export FULL_MIX_JUDGE_MAX_TOKENS=${FULL_MIX_JUDGE_MAX_TOKENS:-4096}
export FULL_MIX_JUDGE_FORCE_JSON=${FULL_MIX_JUDGE_FORCE_JSON:-1}
export FULL_MIX_JUDGE_DISABLE_THINKING=${FULL_MIX_JUDGE_DISABLE_THINKING:-1}
export FULL_MIX_JUDGE_CONCURRENCY=${FULL_MIX_JUDGE_CONCURRENCY:-32}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  \
  algorithm.norm_adv_by_std_in_grpo=False \
  \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  +data.gen_batch_size="${GEN_BATCH_SIZE}" \
  +algorithm.filter_groups.enable=True \
  +algorithm.filter_groups.metric="${FILTER_METRIC}" \
  +algorithm.filter_groups.max_num_gen_batches="${MAX_NUM_GEN_BATCHES}" \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  +algorithm.rollout_correction.rollout_is="${ROLLOUT_IS_LEVEL}" \
  algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
  algorithm.rollout_correction.bypass_mode=false \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.rollout_rs_threshold=null \
  +algorithm.rollout_correction.rollout_rs_threshold_lower=null \
  +algorithm.rollout_correction.rollout_token_veto_threshold=null \
  \
  actor_rollout_ref.actor.loss_agg_mode="token-mean" \
  \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=False \
  \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH}" \
  \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.temperature="${TEMP}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="${REWARD_FN_PATH}" \
  reward.custom_reward_function.name="${REWARD_FN_NAME}" \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.total_epochs=3 \
  trainer.rollout_data_dir=$HOME/verl_dumps/full_mix_ifeval_only_rollouts \
  trainer.validation_data_dir=$HOME/verl_dumps/full_mix_ifeval_only_val \
  "$@"
