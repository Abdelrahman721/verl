set -x

# SNOMED label index for instruction_following / sct_if reward (evaluate_row, label validation).
# Default: sibling checkout of mcs-instruct-dataset next to verl. Override in container if your mount differs.
export INSTRUCTION_FOLLOWING_LABELS_CSV="/workspace/verl/Datasets/set_B1_sct_id_labels.csv"
export WANDB_RUN_ID=uuweq43f      # the run id from wandb/run-...-uuweq43f
export WANDB_RESUME=must         # or "allow"

echo "=== env (instruction_following / judges / ssl) ==="
echo "INSTRUCTION_FOLLOWING_LABELS_CSV=$INSTRUCTION_FOLLOWING_LABELS_CSV"
echo "IF_USE_LLM_JUDGES=$IF_USE_LLM_JUDGES"
echo "DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY"
echo "SSL_CERT_FILE=$SSL_CERT_FILE"
echo "REQUESTS_CA_BUNDLE=$REQUESTS_CA_BUNDLE"
echo "CURL_CA_BUNDLE=$CURL_CA_BUNDLE"
echo "SSL_CERT_DIR=${SSL_CERT_DIR:-}"

# Parquets from examples/data_preprocess/snomed_mixed_combine.py (data_source=sct_mixed, per-row reward dispatch).
MODEL_PATH="models/Qwen3-4B-Base-sft-2345"
TRAIN_FILES="/workspace/verl/rl-data/sct_mixed_train_all.parquet"
VAL_FILES="/workspace/verl/rl-data/sct_mixed_test_all.parquet"

PROJECT_NAME="RL-Exps"
EXP_NAME="grpo++_sct_mixed_all_if_sl_ml"

MAX_PROMPT_LEN=4096
MAX_RESPONSE_LEN=8192

TEMP=1.0
TOP_P=1.0

N_SAMPLES_PER_PROMPT=8

TRAIN_BATCH_SIZE=32
GEN_BATCH_SIZE=96

CLIP_LOW=0.2
CLIP_HIGH=0.28

ROLLOUT_IS_LEVEL="token"
ROLLOUT_IS_THRESHOLD=2.0

FILTER_METRIC="seq_reward"
MAX_NUM_GEN_BATCHES=10

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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.temperature="${TEMP}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=5 \
  trainer.rollout_data_dir=/workspace/verl/verl_dumps/rollouts \
  trainer.validation_data_dir=/workspace/verl/verl_dumps/val \
  "$@"
