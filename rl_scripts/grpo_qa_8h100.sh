set -x

# ===== Model =====
MODEL_PATH="/home/hazem/serve/no-paraphrase"
TRAIN_FILES="/workspace/verl/data/medical_qa_rl/train.parquet"
VAL_FILES="/workspace/verl/data/medical_qa_rl/val.parquet"

PROJECT_NAME="MedQA-RL"
EXP_NAME="no-paraphrase-grpo-8h100"

# ===== Sequence lengths =====
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=8192

# ===== Sampling =====
TEMP=1.0
TOP_P=1.0
N_SAMPLES_PER_PROMPT=16

# ===== Batch sizes =====
# 16 unique prompts × 16 samples = 256 total samples per training step
# 4934 train questions / 16 = ~308 steps per epoch
TRAIN_BATCH_SIZE=4
GEN_BATCH_SIZE=12

# ===== PPO/GRPO clipping =====
CLIP_LOW=0.2
CLIP_HIGH=0.28

# ===== Importance sampling =====
ROLLOUT_IS_LEVEL="token"
ROLLOUT_IS_THRESHOLD=2.0

# ===== Filtering =====
FILTER_METRIC="seq_reward"
MAX_NUM_GEN_BATCHES=10

# ===== WandB =====
export WANDB_API_KEY=""

# ===== Vertex AI judge env (reward function reads these) =====
export QA_JUDGE_CREDENTIALS_FILE="/data/hazem/creds/vertex-sa.json"
export QA_JUDGE_BASE_URL="https://aiplatform.googleapis.com/v1beta1/projects/project-a8ff85a9-571d-4e15-841/locations/global/endpoints/openapi/"
export QA_JUDGE_MODEL="gemini-3-flash-preview"

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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
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
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.use_kl_in_reward=True \
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
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  \
  custom_reward_function.path="/workspace/verl/verl/utils/reward_score/qa.py" \
  custom_reward_function.name="compute_score" \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.save_freq=40 \
  trainer.test_freq=20 \
  trainer.total_epochs=1 \
  trainer.rollout_data_dir=/data/hazem/verl_dumps/rollouts \
  trainer.validation_data_dir=/data/hazem/verl_dumps/val \
  "$@"
