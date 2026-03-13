#!/bin/bash
# Interactive srun: same container setup as verl_sbatch.sh, then drops you into a shell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[[ -f "$SCRIPT_DIR/dev.env" ]] && source "$SCRIPT_DIR/dev.env"

BASE="/data/muhsen/verl-rl"
sudo mkdir -p "$BASE/.cache/huggingface" "$BASE/.cache/torch" "$BASE/.cache/vllm"
export WANDB_API_KEY="${WANDB_API_KEY}"

MOUNTS="${REPO_ROOT}:/workspace/verl"
MOUNTS="${MOUNTS},$BASE/.cache/huggingface:/root/.cache/huggingface,$BASE/.cache/torch:/root/.cache/torch,$BASE/.cache/vllm:/root/.cache/vllm"

srun \
  --partition=cluster \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:8 \
  --cpus-per-task=192 \
  --mem=512G \
  --pty \
  --container-image="verlai/verl:vllm012.latest" \
  --container-mounts="${MOUNTS}" \
  --container-workdir="/workspace/verl" \
  bash -c 'pip3 install --no-deps -e . 2>/dev/null; exec bash -l'
