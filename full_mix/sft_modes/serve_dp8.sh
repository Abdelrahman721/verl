#!/usr/bin/env bash
# vLLM server for dual-mode SFT data generation. DP=8, one engine per GPU.
#
# IMPORTANT: this serves the model's OWN chat template (the minimal ChatML one
# shipped in the checkpoint dir), which renders the generation prompt as
# '<|im_start|>assistant\n' — exactly what the model was RL'd under, so it opens
# its own <think> block. Do NOT serve chat_template_dual_mode.jinja here; that
# template is for rendering the SFT targets, and using it for generation would
# prepend an empty think block and suppress reasoning.
#
# Run INSIDE the container (dev/dev.sh), leave it running, then drive it with
# full_mix/sft_modes/generate_responses.py from another shell.

set -xeuo pipefail

export VLLM_USE_V1=1

MODEL=${MODEL:-/data/abdelrahman/verl/checkpoints/RL-Exps/chat-ifeval-sync-4b/global_step_60/merged_hf_model}
PORT=${PORT:-8300}   # 8000 and 8200 are already taken on this host
DP=${DP:-8}
TP=${TP:-1}

# Prompts are short (p99 ~1.2k); the budget is almost entirely response tokens.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-18432}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
# KV cache is 147 KB/token for this model, so ~410k tokens per engine at 0.90.
# Keep MAX_NUM_SEQS * MAX_MODEL_LEN near that or vLLM preempts with recompute.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-24}

exec vllm serve "${MODEL}" \
  --served-model-name "${MODEL}" \
  --port "${PORT}" \
  --data-parallel-size "${DP}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-prefix-caching \
  "$@"
