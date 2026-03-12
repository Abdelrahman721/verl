#!/bin/bash
#SBATCH --job-name=medqa-grpo
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=/home/hazem/verl-rl/train-%j.log
#SBATCH --error=/home/hazem/verl-rl/train-%j.log

# ── Container config ──
# Uses pyxis to run inside the verl Docker image.
# The verl repo is bind-mounted into the container at /workspace/verl.
# The model dir and HF cache are also mounted.
CONTAINER_IMAGE="verlai/verl:vllm012.latest"
VERL_REPO="/home/hazem/verl"
MODEL_DIR="/home/hazem/serve/no-paraphrase"

# ── Environment variables passed into the container ──
# Set QA_JUDGE_API_KEY before submitting: export QA_JUDGE_API_KEY=...
# Or uncomment and hardcode below:
# export QA_JUDGE_API_KEY="your-key-here"

export WANDB_API_KEY="${WANDB_API_KEY}"

# Ensure cache dirs exist on the node before mounting
mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/torch"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${VERL_REPO}:/workspace/verl,${MODEL_DIR}:${MODEL_DIR},$HOME/.cache/huggingface:/root/.cache/huggingface,$HOME/.cache/torch:/root/.cache/torch" \
  --container-workdir="/workspace/verl" \
  bash -c '
    # Install verl in editable mode (fast, no-deps)
    pip3 install --no-deps -e . 2>/dev/null

    # Run training
    bash rl_scripts/grpo_qa_8h100.sh
  '
