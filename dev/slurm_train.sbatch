#!/bin/bash
#SBATCH --job-name=medqa-grpo
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --output=/data/hazem/logs/train-%j.log
#SBATCH --error=/data/hazem/logs/train-%j.log

# ── Container config ──
# Uses pyxis to run inside the verl Docker image.
# The verl repo is bind-mounted into the container at /workspace/verl.
# The model dir and HF/torch caches are also mounted.
CONTAINER_IMAGE="verlai/verl:vllm012.latest"
VERL_REPO="/data/hazem/verl"
MODEL_DIR="/data/hazem/serve/no-paraphrase"
CACHE_DIR="/data/hazem/.cache"

# ── Environment variables passed into the container ──
# Set QA_JUDGE_API_KEY before submitting: export QA_JUDGE_API_KEY=...
# Or uncomment and hardcode below:
# export QA_JUDGE_API_KEY="your-key-here"

export WANDB_API_KEY="${WANDB_API_KEY}"

# Ensure cache and log dirs exist on the node before mounting
mkdir -p "${CACHE_DIR}/huggingface" "${CACHE_DIR}/torch" "/data/hazem/logs"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${VERL_REPO}:/workspace/verl,${MODEL_DIR}:${MODEL_DIR},${CACHE_DIR}/huggingface:/root/.cache/huggingface,${CACHE_DIR}/torch:/root/.cache/torch" \
  --container-workdir="/workspace/verl" \
  bash -c '
    # Install verl in editable mode + openai for judge
    pip3 install --no-deps -e . 2>/dev/null
    pip3 install openai 2>/dev/null

    # Run training
    bash rl_scripts/grpo_qa_8h100.sh
  '
