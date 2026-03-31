#!/bin/bash
#SBATCH --job-name=rl-mixed
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=/data/muhsen/sbatch/logs/train-%j.log
#SBATCH --error=/data/muhsen/sbatch/logs/train-%j.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="/data/muhsen/repos/verl"
[[ -f "$SCRIPT_DIR/dev.env" ]] && source "$SCRIPT_DIR/dev.env"

BASE="/data/muhsen/verl-rl"
mkdir -p "$BASE/.cache/huggingface" "$BASE/.cache/torch" "$BASE/.cache/vllm"
export WANDB_API_KEY="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"
# mkdir -p /data/muhsen/enroot/{data,runtime,cache}
# export ENROOT_DATA_PATH=/data/muhsen/enroot/data
# export ENROOT_RUNTIME_PATH=/data/muhsen/enroot/runtime
# export ENROOT_CACHE_PATH=/data/muhsen/enroot/cache

MOUNTS="${REPO_ROOT}:/workspace/verl"
MOUNTS="${MOUNTS},$BASE/.cache/huggingface:/root/.cache/huggingface,$BASE/.cache/torch:/root/.cache/torch,$BASE/.cache/vllm:/root/.cache/vllm"

srun \
  --container-image="verlai/verl:vllm012.latest" \
  --container-mounts="${MOUNTS}" \
  --container-workdir="/workspace/verl" \
  bash -c 'pip3 install --no-deps -e . 2>/dev/null; exec bash /workspace/verl/rl_scripts/modified_grpo1.sh'
