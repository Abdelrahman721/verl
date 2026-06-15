#!/usr/bin/env bash
set -euo pipefail

# Merge sharded FSDP actor checkpoints into HuggingFace format.
# Each merged model is written to the checkpoint dir (the source path minus the trailing "actor").

ACTOR_DIRS=(
    "/data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh/global_step_100/actor"
)

for actor_dir in "${ACTOR_DIRS[@]}"; do
    target_dir="$(dirname "$actor_dir")/merged_hf_model"

    echo "=============================================================="
    echo "Merging: $actor_dir"
    echo "Target : $target_dir"
    echo "=============================================================="

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target_dir"
done

echo "All checkpoints merged successfully."
