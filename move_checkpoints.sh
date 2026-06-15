#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh"
DEST_DIR="/scratch/RL/RL-Exps/gpt"

STEPS=(220 320 400 420 440 460 480)

mkdir -p "$DEST_DIR"

for step in "${STEPS[@]}"; do
    src="$SRC_DIR/global_step_$step"
    if [[ -d "$src" ]]; then
        echo "Moving $src -> $DEST_DIR/"
        mv "$src" "$DEST_DIR/"
    else
        echo "WARNING: $src does not exist, skipping" >&2
    fi
done

echo "Done."
