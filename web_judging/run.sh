#!/usr/bin/env bash
# =============================================================================
# web_judging — launch script.
#
# Serves the medical-QA policy checkpoint with vLLM (tensor-parallel 2) and the
# FastAPI web app that lets you: browse train/val entries, generate an answer,
# and grade it with TWO qa-judge backends (Sonnet via Bedrock vs gpt-5.4-mini
# via OpenRouter) sharing ONE Deepseek penalty run.
#
# ALL secret values below are PLACEHOLDERS — fill them in before running.
# Nothing here reads a .env file or any real credential store.
#
# Usage:
#   bash run.sh              # launch vLLM (if LAUNCH_VLLM=1) + the web app
#   LAUNCH_VLLM=0 bash run.sh # web app only; point POLICY_API_BASE at a running vLLM
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1

# ---- policy / vLLM ----------------------------------------------------------
export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh/global_step_380/merged_hf_model}"
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-medical-qa}"
export VLLM_PORT="${VLLM_PORT:-8001}"
export POLICY_API_BASE="${POLICY_API_BASE:-http://127.0.0.1:${VLLM_PORT}/v1}"
export POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"   # vLLM ignores it; client needs non-empty
export POLICY_MAX_LEN="${POLICY_MAX_LEN:-32768}"
export POLICY_TP="${POLICY_TP:-2}"                 # tensor-parallel size (requirement: 2)
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# generation sampling — defaults mirror medical-qa *validation* settings
export GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}"
# export GEN_TOP_P="${GEN_TOP_P:-0.7}"
export GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-16384}"

# ---- qa-judge length penalty (over-length vs gold) --------------------------
# Penalize once the candidate answer exceeds THRESHOLD× the gold word count;
# linear ramp K·(ratio−THRESHOLD), capped at MAX. Applied by BOTH qa judges.
# Lowered from the upstream default of 2.0 → 1.3 so verbosity is penalized sooner.
export QA_LEN_PENALTY_THRESHOLD="${QA_LEN_PENALTY_THRESHOLD:-1.3}"
export QA_LEN_PENALTY_K="${QA_LEN_PENALTY_K:-0.3}"        # ramped up from 0.1 (3× steeper slope)
export QA_LEN_PENALTY_MAX="${QA_LEN_PENALTY_MAX:-0.5}"    # cap raised from 0.3

# ---- System A: qa_bedrock judge (Sonnet via Anthropic Bedrock) --------------
export QA_JUDGE_AWS_ACCESS_KEY="${QA_JUDGE_AWS_ACCESS_KEY:-PLACEHOLDER_AWS_ACCESS_KEY}"
export QA_JUDGE_AWS_SECRET_KEY="${QA_JUDGE_AWS_SECRET_KEY:-PLACEHOLDER_AWS_SECRET_KEY}"
export QA_JUDGE_AWS_REGION="${QA_JUDGE_AWS_REGION:-PLACEHOLDER_AWS_REGION}"        # e.g. eu-central-1
export QA_JUDGE_MODEL="${QA_JUDGE_MODEL:-eu.anthropic.claude-sonnet-4-6}"

# ---- System B: gpt-5.4-mini qa judge (OpenRouter) ---------------------------
export QA_OPENROUTER_API_BASE="${QA_OPENROUTER_API_BASE:-https://openrouter.ai/api/v1}"
export QA_OPENROUTER_API_KEY="${QA_OPENROUTER_API_KEY:-PLACEHOLDER_OPENROUTER_KEY}"
export QA_OPENROUTER_MODEL="${QA_OPENROUTER_MODEL:-openai/gpt-5.4-mini}"

# ---- shared penalty judge (Deepseek via OpenRouter) -------------------------
export FULL_MIX_JUDGE_API_BASE="${FULL_MIX_JUDGE_API_BASE:-https://openrouter.ai/api/v1}"
export FULL_MIX_JUDGE_API_KEY="${FULL_MIX_JUDGE_API_KEY:-PLACEHOLDER_OPENROUTER_KEY}"
export FULL_MIX_JUDGE_MODEL="${FULL_MIX_JUDGE_MODEL:-PLACEHOLDER_DEEPSEEK_MODEL}"  # e.g. deepseek/deepseek-chat
export FULL_MIX_JUDGE_DISABLE_THINKING="${FULL_MIX_JUDGE_DISABLE_THINKING:-0}"     # MUST be 0 for OpenRouter
export FULL_MIX_JUDGE_FORCE_JSON="${FULL_MIX_JUDGE_FORCE_JSON:-1}"
export FULL_MIX_JUDGE_MAX_TOKENS="${FULL_MIX_JUDGE_MAX_TOKENS:-4096}"
export FULL_MIX_JUDGE_MAX_RETRIES="${FULL_MIX_JUDGE_MAX_RETRIES:-3}"
export FULL_MIX_JUDGE_TIMEOUT="${FULL_MIX_JUDGE_TIMEOUT:-120}"
export FULL_MIX_MEDICAL_PENALTY_ENABLED="${FULL_MIX_MEDICAL_PENALTY_ENABLED:-1}"

# ---- data / web -------------------------------------------------------------
export VAL_PARQUET="${VAL_PARQUET:-/data/abdelrahman/verl/data/medical_qa/val.parquet}"
export TRAIN_PARQUET="${TRAIN_PARQUET:-/data/abdelrahman/verl/data/medical_qa/train.parquet}"
export WEB_JUDGING_TRAIN_LIMIT="${WEB_JUDGING_TRAIN_LIMIT:-500}"
export WEB_PORT="${WEB_PORT:-7860}"
# Blind eval: hide the qa-judge model identities (Sonnet vs gpt-5.4-mini) from
# the UI AND the API responses so testers aren't biased. Set to 0 to reveal.
export WEB_JUDGING_BLIND="${WEB_JUDGING_BLIND:-1}"
# Access token: when set, every /api/* request requires it (sent as the
# X-Access-Token header, or once via ?token=... in the URL). Leave as the
# PLACEHOLDER (treated as unset) for no auth on a trusted/local network.
# Share the link as:  http://<host>:<WEB_PORT>/?token=<the-token>
export WEB_JUDGING_TOKEN="${WEB_JUDGING_TOKEN:-aveyolive}"

# ---- launch -----------------------------------------------------------------
LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "$LAUNCH_VLLM" = "1" ]; then
  echo "[run.sh] launching vLLM: $POLICY_MODEL_PATH (TP=$POLICY_TP, port=$VLLM_PORT)"
  vllm serve "$POLICY_MODEL_PATH" \
    --served-model-name "$POLICY_MODEL_NAME" \
    --tensor-parallel-size "$POLICY_TP" \
    --port "$VLLM_PORT" \
    --max-model-len "$POLICY_MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-chunked-prefill \
    > vllm.log 2>&1 &
  VLLM_PID=$!
  echo "[run.sh] vLLM pid=$VLLM_PID, logs -> vllm.log; waiting for it to come up…"
  for i in $(seq 1 120); do
    if curl -sf "${POLICY_API_BASE}/models" >/dev/null 2>&1; then
      echo "[run.sh] vLLM is up."; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[run.sh] vLLM died during startup — see vllm.log"; exit 1
    fi
    sleep 5
  done
else
  echo "[run.sh] LAUNCH_VLLM=0 — expecting a vLLM server already at $POLICY_API_BASE"
fi

echo "[run.sh] starting web app on http://0.0.0.0:${WEB_PORT}"
exec uvicorn server:app --host 0.0.0.0 --port "$WEB_PORT"
