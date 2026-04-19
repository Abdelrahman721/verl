#!/usr/bin/env bash
set -euo pipefail

# Preprocess combined multi-domain datasets for GRPO training.
# Run inside the verl container (see dev/dev.sh).

pip install langdetect immutabledict nltk peft openai

SAVE_DIR="${SAVE_DIR:-/data/abdelrahman/verl/data/combined}"

echo "=============================="
echo " Preprocessing training data  "
echo "=============================="
python3 examples/data_preprocess/combined_train.py \
    --local_save_dir "$SAVE_DIR" \
    ${MATH_LOCAL_PATH:+--math_local_path "$MATH_LOCAL_PATH"} \
    ${IF_LOCAL_PATH:+--if_local_path "$IF_LOCAL_PATH"} \
    ${GENERAL_LOCAL_PATH:+--general_local_path "$GENERAL_LOCAL_PATH"}

echo ""
echo "=============================="
echo " Preprocessing evaluation data"
echo "=============================="
python3 examples/data_preprocess/combined_eval.py \
    --local_save_dir "$SAVE_DIR" \
    ${GSM8K_LOCAL_PATH:+--gsm8k_local_path "$GSM8K_LOCAL_PATH"} \
    ${MATH500_LOCAL_PATH:+--math500_local_path "$MATH500_LOCAL_PATH"} \
    ${IFEVAL_LOCAL_PATH:+--ifeval_local_path "$IFEVAL_LOCAL_PATH"} \
    ${IFBENCH_LOCAL_PATH:+--ifbench_local_path "$IFBENCH_LOCAL_PATH"}

echo ""
echo "=============================="
echo " Done                         "
echo "=============================="
echo "Training parquet : ${SAVE_DIR}/train.parquet"
echo "Eval parquet     : ${SAVE_DIR}/eval.parquet"
