set -x

MODEL_PATH="Qwen/Qwen3-4B"
TRAIN_PARQUET="$HOME/data/gsm8k/train.parquet"
VAL_PARQUET="$HOME/data/gsm8k/test.parquet"

GRPO_GROUP_N=8
PROMPT_BATCH=64

PPO_EPOCHS=1
CLIP_RATIO=0.2

KL_COEF=1e-3
KL_TYPE="kl"

MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=4096

TEMP=1.0
TOP_P=1.0

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size="${PROMPT_BATCH}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PROMPT_BATCH}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_TYPE}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.n="${GRPO_GROUP_N}" \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.temperature="${TEMP}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO}" \
  actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="RL-Exps" \
  trainer.experiment_name="run_naive_grpo_$(date +%Y%m%d_%H%M%S)" \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  trainer.rollout_data_dir=$HOME/verl_dumps/rollouts \
  trainer.validation_data_dir=$HOME/verl_dumps/val \
  trainer.total_epochs=2 $@


