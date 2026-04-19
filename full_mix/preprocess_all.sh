#!/usr/bin/env bash
# Preprocess all four training sets + four eval sets for the full_mix run.
# Run from the verl repo root (or anywhere — paths are absolute).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- source dataset paths (override via env if needed) ----
MATH_SRC=${MATH_SRC:-/data/abdelrahman/qwen-sft/rl-data-prep/data/dolci_rl_zero_math_7b}
IF_SRC=${IF_SRC:-/data/abdelrahman/qwen-sft/rl-data-prep/data/dolci_think_rl_32b_ifeval}
CHAT_SRC=${CHAT_SRC:-/data/abdelrahman/qwen-sft/rl-data-prep/data/dolci_think_rl_32b_chat}
SAFETY_SRC=${SAFETY_SRC:-/data/abdelrahman/qwen-sft/rl-data-prep/data/safety_dpo_with_reference}

# ---- output directories ----
OUT_DIR=${OUT_DIR:-/data/abdelrahman/verl/data/full_mix}
TRAIN_DIR="$OUT_DIR/train"
EVAL_DIR="$OUT_DIR/eval"

# ---- per-domain row caps (optional) ----
MATH_MAX_ROWS=${MATH_MAX_ROWS:-}
IF_MAX_ROWS=${IF_MAX_ROWS:-}
CHAT_MAX_ROWS=${CHAT_MAX_ROWS:-}
SAFETY_MAX_ROWS=${SAFETY_MAX_ROWS:-}

# ---- optional local mirrors for the eval HF datasets ----
GSM8K_LOCAL_PATH=${GSM8K_LOCAL_PATH:-}
MATH500_LOCAL_PATH=${MATH500_LOCAL_PATH:-}
IFEVAL_LOCAL_PATH=${IFEVAL_LOCAL_PATH:-}
IFBENCH_LOCAL_PATH=${IFBENCH_LOCAL_PATH:-}

PY=${PYTHON:-python3}

with_optional() {
  # Emit a flag + value only if the value is non-empty.
  local flag="$1"; local val="$2"
  if [[ -n "$val" ]]; then echo "--${flag}=${val}"; fi
}

echo "[full_mix] ==== TRAIN =========================================="
$PY -m full_mix.preprocess.math_train   --src "$MATH_SRC"   --out_dir "$TRAIN_DIR" $(with_optional max_rows "$MATH_MAX_ROWS")
$PY -m full_mix.preprocess.ifeval_train --src "$IF_SRC"     --out_dir "$TRAIN_DIR" $(with_optional max_rows "$IF_MAX_ROWS")
$PY -m full_mix.preprocess.chat_train   --src "$CHAT_SRC"   --out_dir "$TRAIN_DIR" $(with_optional max_rows "$CHAT_MAX_ROWS")
$PY -m full_mix.preprocess.safety_train --src "$SAFETY_SRC" --out_dir "$TRAIN_DIR" $(with_optional max_rows "$SAFETY_MAX_ROWS")

echo "[full_mix] ==== EVAL ==========================================="
$PY -m full_mix.preprocess.gsm8k_eval    --out_dir "$EVAL_DIR" $(with_optional local_path "$GSM8K_LOCAL_PATH")
$PY -m full_mix.preprocess.math500_eval  --out_dir "$EVAL_DIR" $(with_optional local_path "$MATH500_LOCAL_PATH")
$PY -m full_mix.preprocess.ifeval_eval   --out_dir "$EVAL_DIR" $(with_optional local_path "$IFEVAL_LOCAL_PATH")
$PY -m full_mix.preprocess.ifbench_eval  --out_dir "$EVAL_DIR" $(with_optional local_path "$IFBENCH_LOCAL_PATH")

echo "[full_mix] done; outputs under $OUT_DIR"
