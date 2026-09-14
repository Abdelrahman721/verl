#!/usr/bin/env bash
# Preprocess the general-phase training sets + eval sets for the QLCM pipeline.
# Run from anywhere — paths are absolute.
#
# Math training was dropped from this pipeline (the 32B production run dropped
# it too), but GSM8K and MATH-500 are still built as *eval* sets so the general
# phase keeps a read on reasoning while it trains on IFEval / chat / safety.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- source dataset paths (override via env if needed) ----
IF_SRC=${IF_SRC:-/data/abdelrahman/qwen-sft/rl-data-prep/data/dolci_think_rl_32b_ifeval}
CHAT_SRC=${CHAT_SRC:-/data/abdelrahman/qwen-sft/rl-data-prep/data/dolci_think_rl_32b_chat_with_baseline}
SAFETY_SRC=${SAFETY_SRC:-/data/abdelrahman/qwen-sft/rl-data-prep/data/safety_dpo_with_reference}
IDENTITY_SRC=${IDENTITY_SRC:-/data/abdelrahman/qwen-sft/rl-data-prep/data/identity}

# ---- output directories ----
OUT_DIR=${OUT_DIR:-/data/abdelrahman/verl/data/qlcm}
TRAIN_DIR="$OUT_DIR/train"
EVAL_DIR="$OUT_DIR/eval"

# ---- per-domain row caps (optional) ----
IF_MAX_ROWS=${IF_MAX_ROWS:-}
CHAT_MAX_ROWS=${CHAT_MAX_ROWS:-}
SAFETY_MAX_ROWS=${SAFETY_MAX_ROWS:-}
IDENTITY_MAX_ROWS=${IDENTITY_MAX_ROWS:-}

# ---- optional local mirrors for the eval HF datasets ----
GSM8K_LOCAL_PATH=${GSM8K_LOCAL_PATH:-}
MATH500_LOCAL_PATH=${MATH500_LOCAL_PATH:-}
IFEVAL_LOCAL_PATH=${IFEVAL_LOCAL_PATH:-}

PY=${PYTHON:-python3}

with_optional() {
  # Emit a flag + value only if the value is non-empty.
  local flag="$1"; local val="$2"
  if [[ -n "$val" ]]; then echo "--${flag}=${val}"; fi
}

echo "[qlcm] ==== TRAIN =============================================="
$PY -m qlcm.preprocess.ifeval_train               --src "$IF_SRC"       --out_dir "$TRAIN_DIR" $(with_optional max_rows "$IF_MAX_ROWS")
$PY -m qlcm.preprocess.chat_with_baseline_train   --src "$CHAT_SRC"     --out_dir "$TRAIN_DIR" $(with_optional max_rows "$CHAT_MAX_ROWS")
$PY -m qlcm.preprocess.safety_train               --src "$SAFETY_SRC"   --out_dir "$TRAIN_DIR" $(with_optional max_rows "$SAFETY_MAX_ROWS")
$PY -m qlcm.preprocess.identity_train             --src "$IDENTITY_SRC" --out_dir "$TRAIN_DIR" $(with_optional max_rows "$IDENTITY_MAX_ROWS")

echo "[qlcm] ==== EVAL ==============================================="
$PY -m qlcm.preprocess.gsm8k_eval    --out_dir "$EVAL_DIR" $(with_optional local_path "$GSM8K_LOCAL_PATH")
$PY -m qlcm.preprocess.math500_eval  --out_dir "$EVAL_DIR" $(with_optional local_path "$MATH500_LOCAL_PATH")
$PY -m qlcm.preprocess.ifeval_eval   --out_dir "$EVAL_DIR" $(with_optional local_path "$IFEVAL_LOCAL_PATH")

echo "[qlcm] done; outputs under $OUT_DIR"
echo "[qlcm] next: bash qlcm/run_pipeline.sh"
