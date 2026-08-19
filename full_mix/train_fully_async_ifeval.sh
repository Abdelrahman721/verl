#!/usr/bin/env bash
# Fully-async GRPO training on IFEval ONLY (separate experiment from
# train_fully_async_no_math.sh). Uses the quality-gated IFEval reward
# (full_mix/rewards/compute_score_ifeval.py + ifeval_quality_gated.py) to
# discourage reward hacking — i.e. the model emitting gibberish that
# satisfies constraints without genuinely engaging with the task.
#
# Reuses the same shard layout as the multi-source curriculum:
#   data/full_mix/train/shards/raw/shard_{1..4}/ifeval_train.parquet
# Stage k's training data lives at:
#   data/full_mix/train/shards/ifeval_only/stage_{k}/ifeval_train.parquet
# (created by curriculum_commands_ifeval.sh and build_next_stage_ifeval.py)
#
# Resource model + nearly all knobs are inherited from the multi-source
# script. Differences are tagged with "[IFEVAL]" comments below.
#
# Run INSIDE the container started by dev/dev.sh.

set -xeuo pipefail

export VLLM_USE_V1=1

# _FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# if [[ -f "$_FM_DIR/secrets.env" ]]; then
#   source "$_FM_DIR/secrets.env"
# fi


# ---- model ----
# Stage chaining: same merger workflow as the multi-source experiment
# (no resume; MODEL_PATH points at previous stage's merged_hf).
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}


# ---- data ----  [IFEVAL: only ifeval, only ifeval_eval]
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/full_mix}

STAGE=${STAGE:-1}
if [[ -n "$STAGE" ]]; then
  _STAGE_TRAIN_DIR="${DATA_DIR}/train/shards/ifeval_only/stage_${STAGE}"
  TRAIN_FILES=${TRAIN_FILES:-"[${_STAGE_TRAIN_DIR}/ifeval_train.parquet]"}
else
  TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/train/ifeval_train.parquet]"}
fi
VAL_FILES=${VAL_FILES:-"[${DATA_DIR}/eval/ifeval_eval.parquet]"}


# ---- experiment metadata ----  [IFEVAL: distinct EXP_NAME prefix]
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
if [[ -n "$STAGE" ]]; then
  EXP_NAME=${EXP_NAME:-ifeval-only-stage${STAGE}}
else
  EXP_NAME=${EXP_NAME:-ifeval-only-fresh}
fi


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
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-10}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
GRAD_CLIP=${GRAD_CLIP:-1.0}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}


# ---- overlong buffer ----
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-True}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}


# ---- importance sampling ----
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

# TOTAL_ROLLOUT_STEPS unset => rollouter uses len(dataloader) * TOTAL_EPOCHS
# = dataset_size for one epoch.
if [[ -n "$STAGE" ]]; then
  TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-}
else
  TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-42100}
fi

if [[ -n "$STAGE" ]]; then
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
else
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
fi
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-20}

# Curriculum stages chain via MODEL_PATH (no resume), so RESUME_FROM_PATH is
# normally empty. Kept here for parity with the multi-source script.
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
if [[ -n "$RESUME_FROM_PATH" ]]; then
  RESUME_MODE=${RESUME_MODE:-resume_path}
else
  RESUME_MODE=${RESUME_MODE:-disable}
fi


# ---- rollout / validation dump dirs ----  [IFEVAL: distinct dirs]
if [[ -n "$STAGE" ]]; then
  ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$HOME/verl_dumps/ifeval_only_stage${STAGE}_rollouts}
else
  ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$HOME/verl_dumps/ifeval_only_rollouts}
fi
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$HOME/verl_dumps/ifeval_only_val}


# ---- cluster layout ----
NNODES_ROLLOUT=${NNODES_ROLLOUT:-2}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# ---- parallelism ----
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 3 ))}
GEN_TP=${GEN_TP:-1}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-8}


# ---- offloading ----
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-False}
REF_OFFLOAD=${REF_OFFLOAD:-True}


# ---- vLLM rollout knobs ----
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.70}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}


# ---- reward dispatcher ----  [IFEVAL: dedicated dispatcher]
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/full_mix/rewards/compute_score_ifeval.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}


# ---- judge ----
# Same env knobs as the multi-source script. Anti-hacking gate has its own
# extra knobs:
#   FULL_MIX_IFEVAL_GATE_ENABLED       1/0 (default 1)
#   FULL_MIX_IFEVAL_GATE_THRESHOLD     skip judge below this rule_score (default 0.5)
#   FULL_MIX_IFEVAL_GATE_MULT_GENUINE  default 1.0
#   FULL_MIX_IFEVAL_GATE_MULT_BORDERLINE default 0.3
#   FULL_MIX_IFEVAL_GATE_MULT_GIBBERISH  default 0.0
export FULL_MIX_JUDGE_API_BASE=${FULL_MIX_JUDGE_API_BASE:-http://127.0.0.1:8000/v1}
export FULL_MIX_JUDGE_API_KEY=${FULL_MIX_JUDGE_API_KEY:-EMPTY}
export FULL_MIX_JUDGE_MODEL=${FULL_MIX_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export FULL_MIX_JUDGE_TIMEOUT=${FULL_MIX_JUDGE_TIMEOUT:-120}
export FULL_MIX_JUDGE_MAX_RETRIES=${FULL_MIX_JUDGE_MAX_RETRIES:-10}
export FULL_MIX_JUDGE_MAX_TOKENS=${FULL_MIX_JUDGE_MAX_TOKENS:-4096}
export FULL_MIX_JUDGE_FORCE_JSON=${FULL_MIX_JUDGE_FORCE_JSON:-1}
export FULL_MIX_JUDGE_DISABLE_THINKING=${FULL_MIX_JUDGE_DISABLE_THINKING:-1}
export FULL_MIX_JUDGE_CONCURRENCY=${FULL_MIX_JUDGE_CONCURRENCY:-32}

export FULL_MIX_IFEVAL_GATE_ENABLED=${FULL_MIX_IFEVAL_GATE_ENABLED:-1}
export FULL_MIX_IFEVAL_GATE_THRESHOLD=${FULL_MIX_IFEVAL_GATE_THRESHOLD:-0.5}
export FULL_MIX_IFEVAL_GATE_MULT_GENUINE=${FULL_MIX_IFEVAL_GATE_MULT_GENUINE:-1.0}
export FULL_MIX_IFEVAL_GATE_MULT_BORDERLINE=${FULL_MIX_IFEVAL_GATE_MULT_BORDERLINE:-0.3}
export FULL_MIX_IFEVAL_GATE_MULT_GIBBERISH=${FULL_MIX_IFEVAL_GATE_MULT_GIBBERISH:-0.0}


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
  rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS:-null}" \
  \
  async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
  async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}" \
  async_training.require_batches="${REQUIRE_BATCHES}" \
  async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
  ${RAY_ENV_FLAGS:-} \
  "$@"
