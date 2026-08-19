#!/usr/bin/env bash
# Fully-async (disaggregated) off-policy GRPO for Qwen2.5-32B
#   2 nodes x 8 GPUs  ->  node A = Trainer (8 GPUs), node B = Rollouter (8 GPUs)
#
# Submit from the HEAD node only, after the ray cluster is up:
#   RAY_ADDRESS=http://127.0.0.1:8265 bash rl_scripts/fully_async_qwen32b_2node.sh
set -xeuo pipefail

project_name=${PROJECT_NAME:-'RL-Exps'}
exp_name=${EXP_NAME:-'qwen32b-fully-async'}

RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/experimental/fully_async_policy/shell/runtime_env.yaml"}

# ---- paths (must be identical + readable on BOTH nodes) ----
MODEL_PATH=${MODEL_PATH:-"/data/abdelrahman/models/Qwen2.5-32B"}
TRAIN_FILE=${TRAIN_FILE:-"/data/abdelrahman/verl/data/your_dataset/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/abdelrahman/verl/data/your_dataset/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"/data/abdelrahman/verl/checkpoints/${project_name}/${exp_name}"}
REWARD_FN=${REWARD_FN:-""}   # e.g. /data/abdelrahman/verl/rl_scripts/combined_reward.py

# ---- rollout engine: MUST be vllm in server/AgentLoop mode ----
export VLLM_USE_V1=1
rollout_name="vllm"
rollout_mode="async"
return_raw_chat="True"

# ---- resource split ----
n_nodes_train=1;   n_gpus_training=8
n_nodes_rollout=1; n_gpus_rollout=8

# ---- lengths ----
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

# ---- algorithm ----
adv_estimator=grpo
use_kl_in_reward=False       # False => no ref policy worker allocated at all (saves memory)
use_kl_loss=False
kl_coef=0.0
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# ---- async schedule ----
n_resp_per_prompt=8          # GRPO group size
train_prompt_mini_bsz=32     # prompts per optimizer update (global, not per-GPU)
require_batches=2            # trainer pulls require_batches * mini_bsz prompts per fetch
trigger_parameter_sync_step=4    # local updates between NCCL weight syncs
# => effective "train_batch_size" per policy version = 32 * 2 * 4 = 256 prompts
effective_train_bsz=$((train_prompt_mini_bsz * require_batches * trigger_parameter_sync_step))
staleness_threshold=0.3      # 0 = sync; >0 = off-policy. keep < 1
partial_rollout=True         # interrupt+resume in-flight rollouts across weight syncs
total_rollout_steps=$((effective_train_bsz * 400))   # ~400 policy versions
test_freq=25                 # validate every N weight syncs
save_freq=25

# ---- fully-async requires streaming data feed ----
train_prompt_bsz=0
gen_prompt_bsz=1

# ---- perf / memory (32B dense on 8 GPUs) ----
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
sp_size=4                    # ulysses sequence parallel on the trainer
fsdp_size=-1                 # shard params over all 8 trainer GPUs
actor_offload=True           # 32B + Adam does not fit 8x80GB without offload
gen_tp=4                     # vLLM TP=4 -> 2 DP replicas on the rollout node
nccl_timeout=72000

EXTRA_ARGS=()
if [[ -n "${REWARD_FN}" ]]; then
  EXTRA_ARGS+=( "reward.custom_reward_function.path=${REWARD_FN}" )
fi

ray job submit --no-wait \
  --address "${RAY_ADDRESS}" \
  --runtime-env="${RUNTIME_ENV}" \
  --working-dir "${WORKING_DIR}" \
  -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.truncation='left' \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.train_batch_size=${train_prompt_bsz} \
  data.gen_batch_size=${gen_prompt_bsz} \
  data.return_raw_chat=${return_raw_chat} \
  \
  algorithm.adv_estimator=${adv_estimator} \
  algorithm.use_kl_in_reward=${use_kl_in_reward} \
  algorithm.kl_ctrl.kl_coef=${kl_coef} \
  algorithm.rollout_correction.bypass_mode=True \
  \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.nccl_timeout=${nccl_timeout} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
  actor_rollout_ref.actor.use_rollout_log_probs=True \
  actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
  actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
  critic.strategy=fsdp2 \
  \
  actor_rollout_ref.rollout.name=${rollout_name} \
  actor_rollout_ref.rollout.mode=${rollout_mode} \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  \
  trainer.nnodes=${n_nodes_train} \
  trainer.n_gpus_per_node=${n_gpus_training} \
  rollout.nnodes=${n_nodes_rollout} \
  rollout.n_gpus_per_node=${n_gpus_rollout} \
  rollout.total_rollout_steps=${total_rollout_steps} \
  \
  async_training.staleness_threshold=${staleness_threshold} \
  async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
  async_training.require_batches=${require_batches} \
  async_training.partial_rollout=${partial_rollout} \
  async_training.use_trainer_do_validate=False \
  \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.val_before_train=True \
  trainer.test_freq=${test_freq} \
  trainer.save_freq=${save_freq} \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode=auto \
  trainer.total_epochs=10 \
  "${EXTRA_ARGS[@]}"
