#!/bin/bash
#SBATCH --job-name=medqa-grpo-mn
#SBATCH --partition=cluster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=/home/hazem/verl-rl/train-mn-%j.log
#SBATCH --error=/home/hazem/verl-rl/train-mn-%j.log

# ── Container config ──
CONTAINER_IMAGE="verlai/verl:vllm012.latest"
VERL_REPO="/home/hazem/verl"
MODEL_DIR="/home/hazem/serve/no-paraphrase"

# ── Environment ──
export WANDB_API_KEY="${WANDB_API_KEY}"

# Ensure cache dirs exist on all nodes before mounting
srun --ntasks-per-node=1 bash -c 'mkdir -p $HOME/.cache/huggingface $HOME/.cache/torch'

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${VERL_REPO}:/workspace/verl,${MODEL_DIR}:${MODEL_DIR},$HOME/.cache/huggingface:/root/.cache/huggingface,$HOME/.cache/torch:/root/.cache/torch" \
  --container-workdir="/workspace/verl" \
  bash -c '
    pip3 install --no-deps -e . 2>/dev/null

    bash rl_scripts/grpo_qa_multinode.sh
  '
