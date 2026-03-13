#!/bin/bash
#SBATCH --job-name=snomed-grpo
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=512G
#SBATCH --output=/data/muhsen/verl-rl/train-%j.log
#SBATCH --error=/data/muhsen/verl-rl/train-%j.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[[ -f "$SCRIPT_DIR/dev.env" ]] && source "$SCRIPT_DIR/dev.env"

BASE="/data/muhsen/verl-rl"
mkdir -p "$BASE/.cache/huggingface" "$BASE/.cache/torch" "$BASE/.cache/vllm"
export WANDB_API_KEY="${WANDB_API_KEY}"

MOUNTS="${REPO_ROOT}:/workspace/verl"
MOUNTS="${MOUNTS},$BASE/.cache/huggingface:/root/.cache/huggingface,$BASE/.cache/torch:/root/.cache/torch,$BASE/.cache/vllm:/root/.cache/vllm"

srun \
  --container-image="verlai/verl:vllm012.latest" \
  --container-mounts="${MOUNTS}" \
  --container-workdir="/workspace/verl" \
  bash -c 'pip3 install --no-deps -e . 2>/dev/null; exec bash rl_scripts/modified_grpo.sh'
