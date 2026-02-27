#!/bin/bash
set -x

# Model and Data Configuration
MODEL_PATH="Qwen/Qwen3-4B-Base"
TRAIN_FILES="/workspace/verl/data/train_set_sft_trajectories_a.parquet"
VAL_FILES="/workspace/verl/data/train_set_sft_trajectories_a.parquet"

# Training Configuration
PROJECT_NAME="ICD-SFT"
EXP_NAME="qwen3-4b-sft-full"

# Training Parameters
TRAIN_BATCH_SIZE=8
MICRO_BATCH_SIZE_PER_GPU=8
MAX_LENGTH=2048
NUM_GPUS=1

# Optimization Parameters
LR=2e-5
WEIGHT_DECAY=0.01
LR_WARMUP_RATIO=0.1
CLIP_GRAD=1.0

# Training Epochs and Checkpointing
TOTAL_EPOCHS=5
SAVE_FREQ=1400  # -1 means save at end of each epoch
TEST_FREQ=-1  # -1 means test at end of each epoch

# Data Configuration
PROMPT_KEY="input"
RESPONSE_KEY="output"
TRUNCATION="right"

# Checkpoint Configuration
CKPTS_HOME="$HOME/verl_checkpoints/"
mkdir -p "${CKPTS_HOME}"

# Export WANDB API Key (optional)
export WANDB_API_KEY="7eadd40652b0651b0f12dc86ea4d5fde56db2e2a"

echo "Starting SFT training for ${MODEL_PATH} on ${TRAIN_FILES}"
echo "Training will run for ${TOTAL_EPOCHS} epochs with checkpointing at each epoch"
echo "Training will be saved to: ${CKPTS_HOME}"

# Run SFT training using FSDP trainer (epoch-based with checkpointing at each epoch)
torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key="${PROMPT_KEY}" \
    data.response_key="${RESPONSE_KEY}" \
    data.max_length="${MAX_LENGTH}" \
    data.truncation="${TRUNCATION}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    \
    model.partial_pretrain="${MODEL_PATH}" \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.model_dtype=bfloat16 \
    model.strategy=fsdp2 \
    model.lora_rank=0 \
    model.trust_remote_code=True \
    \
    optim.lr="${LR}" \
    optim.weight_decay="${WEIGHT_DECAY}" \
    optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO}" \
    optim.clip_grad="${CLIP_GRAD}" \
    optim.lr_scheduler=cosine \
    optim.betas="[0.9,0.95]" \
    \
    trainer.default_local_dir="${CKPTS_HOME}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps=null \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.logger='["console","wandb"]' \
    trainer.checkpoint.save_contents='[model,optimizer,hf_model]' \
    trainer.max_ckpt_to_keep=5 \
    \
    use_remove_padding=True \
    "$@"

echo "SFT training completed!"
echo "Model saved to: ${CKPTS_HOME}"