#!/usr/bin/env bash
# One-time dependency install inside the dev container (dev/dev.sh).
# Run on EVERY node before `ray start`.

set -euo pipefail

pip install langdetect immutabledict nltk peft openai liger-kernel
pip install "cupy-cuda12x<14"
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# All environment configuration — judge credentials, models, provider pinning,
# wandb — lives in ONE file: qlcm/runtime_env.yaml (gitignored).
#
# qlcm/load_runtime_env.sh loads it into every training script AND forwards the
# resolved values to the Ray actors, so the same file works whether you run the
# scripts directly or submit them with `ray job submit --runtime-env=...`.
# It only needs to exist on the node you launch from.

echo "next:"
echo "  cp qlcm/runtime_env.yaml.example qlcm/runtime_env.yaml   # then fill in the keys"
echo "  bash qlcm/preprocess_all.sh"
echo "  bash qlcm/run_pipeline.sh"
