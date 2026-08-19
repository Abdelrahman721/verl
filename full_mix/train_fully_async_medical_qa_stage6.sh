#!/usr/bin/env bash
# Fully-async GRPO training on the Medical QA STAGE-6 dataset (+retention).
# Multilabel-only focus on REAL clinical-note coding data — ICD-10
# (multilabel + single-label) and SNOMED CT (multilabel), concatenated with a
# stage-5-aware retention pool.
#
# Self-contained experiment. Differences from stage-5:
#   - Datasets:       data/medical_qa_stage6/train.parquet     (3 378 rows; per-source duplication for epochs)
#                     data/medical_qa_stage6/retention.parquet (2 013 rows)
#                     data/medical_qa_stage6/val.parquet       (681 rows, take-all — no subsample)
#                       Both train parquets share a unified 39-field
#                       extra_info struct (same shape as stage-5) so HF
#                       datasets can concat-load them in one shot.
#
#                       train.parquet pools (3 378, in-parquet epoch duplication):
#                         mlb (avey_mlb_train, real ICD multilabel)        308 × 3 =   924
#                         slb (avey_icd_slb_rl_train, real ICD SL)       2 145 × 1 = 2 145
#                         snomed_ml (final_ml_real, real SNOMED multilabel) 103 × 3 =   309
#                       All three stamped:
#                         mlb / slb       → task_type=icd10_multilabel    → icd_scorer
#                         snomed_ml       → task_type=snomed_multilabel   → snomed_ml_scorer
#
#                       retention.parquet pools (2 013, seed=6051):
#                         general (chat/ifeval/safety, safety=50%)             405
#                         medical_qa UNSEEN                                    608
#                         stage-5 coding UNMASTERED (cap 1 000 of 1 898)     1 000
#
#                       val.parquet pools (681, take-all):
#                         mlb         77
#                         slb        536
#                         snomed_ml   68
#
#   - Reward fn:      full_mix/rewards/compute_score_coding_mix.py
#                       Three-branch top-level dispatcher:
#                         CODING — data_source ∈ {ml, if, snomed_ml, snomed_if,
#                                                  sl, snomed_sl}
#                             -> full_mix.rewards.coding.compute_score
#                                Sub-routes on extra_info.task_type:
#                                  icd10_multilabel +
#                                  icd10_instruction_follow +
#                                  icd_multi_label              -> icd_scorer
#                                  snomed_multilabel +
#                                  snomed_single_label          -> snomed_ml_scorer
#                                  snomed_instruction_follow    -> snomed_if_scorer
#                                                                  (uses the embedded
#                                                                   instruction_following framework)
#                         MEDICAL/BENCH — data_source ∈ {medical_qa, medical_conv,
#                                                        medical_benchmark_*}
#                             -> full_mix.rewards.qa_openrouter_bench.compute_score
#                                OpenRouter judge for qa + conversation,
#                                deterministic bench scorers otherwise.
#                         GENERAL RETENTION — any other data_source
#                             -> full_mix.rewards.compute_score.compute_score
#                                Existing ifeval/chat/safety/identity dispatcher.
#                       All output reward dicts are normalised to the 29-key
#                       triple-union (15 coding + 23 qa_openrouter_bench + 2
#                       retention sentinels, deduped).
#                       Stage-6 expands the coding suite's accepted data_sources
#                       to include `mlb` and `slb` (real ICD), both stamped
#                       task_type=icd10_multilabel and routed to the ICD scorer.
#   - Judges:         Stage-6 train is multilabel-only, so the SNOMED IF scorer
#                       is not exercised by the train rows. The retention pool
#                       brings in stage-5 IF-unmastered prompts (~1 000 rows
#                       including 681 ICD-IF + 316 SNOMED-IF), so the IF judge
#                       block still applies during retention scoring. Gated by
#                       IF_USE_LLM_JUDGES; reads OPENROUTER_API_KEY from secrets.env.
#                       The labels JSON (sct_id -> description) ships INSIDE
#                       the coding package — no override needed.
#
# Inherits cluster layout, sampling, KL-in-reward regime, batching, overlong
# buffer, PPO clip / loss, vLLM, FSDP and offload from the stage-5 runner.
#
# Run INSIDE the container started by dev/dev.sh.

set -xeuo pipefail

export VLLM_USE_V1=1

# _FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# if [[ -f "$_FM_DIR/secrets.env" ]]; then
#   source "$_FM_DIR/secrets.env"
# fi


# ---- model ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}


# ---- data ----
# Two train parquets (coding + retention), one val parquet. verl loads them
# via HF datasets which concat-loads multiple files into a single dataset.
# Both train parquets MUST share the same Arrow schema — the build pipeline
# harmonises extra_info to a 39-field union and normalises the prompt struct
# field order so they concat cleanly. If you swap in different parquets,
# re-run the harmonisation pass.
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data}
TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/medical_qa_stage6/train.parquet,${DATA_DIR}/medical_qa_stage6/retention.parquet]"}
VAL_FILES=${VAL_FILES:-"[${DATA_DIR}/medical_qa_stage6/val.parquet]"}


# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
EXP_NAME=${EXP_NAME:-medical-qa-stage6-coding}


# ---- sequence lengths ----
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-8192}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}


# ---- sampling ----
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-0.7}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}


# ---- GRPO / KL knobs ----
# Same KL-IN-REWARD regime as stage-4 (token-level KL penalty added into the
# reward signal at a small coefficient, NOT KL-in-loss).
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-True}
KL_COEF=${KL_COEF:-0.0001}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}


# ---- PPO clip / loss / optim ----
CLIP_LOW=${CLIP_LOW:-0.2}
CLIP_HIGH=${CLIP_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
LR=${LR:-1e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-20}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
GRAD_CLIP=${GRAD_CLIP:-1.0}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}


# ---- overlong buffer (DAPO-style soft penalty near the response-length cap) ----
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-True}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}


# ---- importance sampling (off-policy correction) ----
USE_ROLLOUT_LOG_PROBS=${USE_ROLLOUT_LOG_PROBS:-True}


# ---- async-training knobs ----
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.25}
TRIGGER_SYNC_STEP=${TRIGGER_SYNC_STEP:-4}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}


# ---- batching ----
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-0}
GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-1}
TRAIN_MINI_BSZ=${TRAIN_MINI_BSZ:-16}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-1000000}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-20}

# Resume policy:
#   RESUME_FROM_PATH non-empty -> RESUME_MODE=resume_path, pass the path through.
#   RESUME_FROM_PATH empty     -> RESUME_MODE=disable (start from MODEL_PATH).
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
if [[ -n "$RESUME_FROM_PATH" ]]; then
  RESUME_MODE=${RESUME_MODE:-resume_path}
else
  RESUME_MODE=${RESUME_MODE:-disable}
fi


# ---- rollout / validation dump dirs ----
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$HOME/verl_dumps/medical_qa_stage6_rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$HOME/verl_dumps/medical_qa_stage6_val}


# ---- cluster layout (rollout and trainer on DISJOINT nodes) ----
NNODES_ROLLOUT=${NNODES_ROLLOUT:-3}
NNODES_TRAIN=${NNODES_TRAIN:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# ---- parallelism (dynamic-bsz + Ulysses SP + vLLM TP) ----
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$((( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 3 ) / 4))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
GEN_TP=${GEN_TP:-4}
SP_SIZE=${SP_SIZE:-8}
FSDP_SIZE=${FSDP_SIZE:-16}


# ---- offloading ----
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-True}
REF_OFFLOAD=${REF_OFFLOAD:-True}


# ---- vLLM rollout knobs ----
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}


# ---- reward dispatcher ----
# Stage-5 = coding only (no retention this stage). The compute_score_coding_mix
# dispatcher already supports both branches; the retention branch is just
# unused for now. When retention is added, drop the parquet sources into
# TRAIN_FILES and they'll route through full_mix.rewards.compute_score.
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/full_mix/rewards/compute_score_coding_mix.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}


# ---- instruction-following env (snomed_if scorer) ----
# Embedded labels JSON is the default; only set this if you want to override.
# export INSTRUCTION_FOLLOWING_LABELS_CSV=/path/to/all_codes_to_descriptions_reconciled.json
#
# Enable DeepSeek (via OpenRouter) judging for the v08/v09/v11/v12 rules.
# Set to 0 to keep the IF scorer deterministic (the other 13 rules don't
# use the judge regardless).
export IF_USE_LLM_JUDGES=${IF_USE_LLM_JUDGES:-1}


# ---- judges (medical + retention branches) ----
# Three branches now consult external judges:
#   * Coding/SNOMED-IF       → DeepSeek via OpenRouter (uses OPENROUTER_API_KEY)
#   * Medical/bench (qa_*)   → OpenRouter (reads QA_JUDGE_*)
#   * Retention (chat/safety/identity from full_mix.compute_score)
#                            → local vLLM judge OR a remote endpoint (reads FULL_MIX_JUDGE_*)
# All keys live in secrets.env (gitignored).

# Shared OpenRouter key — used by both the IF scorer AND qa_openrouter_bench.
export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}

# qa_openrouter_bench (medical_qa / medical_conv / medical_benchmark_* rows)
# Resolution order inside the module:
#   1. QA_JUDGE_OPENROUTER_API_KEY (judge-specific override)
#   2. OPENROUTER_API_KEY          (shared)
export QA_JUDGE_OPENROUTER_API_KEY=${QA_JUDGE_OPENROUTER_API_KEY:-${OPENROUTER_API_KEY:-}}
# Optional proxy / self-host base URL — default is https://openrouter.ai/api/v1.
# export QA_JUDGE_OPENROUTER_BASE_URL=...
export QA_JUDGE_MODEL=${QA_JUDGE_MODEL:-openai/gpt-5.4-mini}
export QA_JUDGE_MAX_TOKENS=${QA_JUDGE_MAX_TOKENS:-32768}

# Full-mix retention judges (chat / safety / identity / global identity gate).
# Used for the general-retention pool in retention.parquet (local/dolci-*,
# local/safety-dpo-reference, etc.).
export FULL_MIX_JUDGE_API_BASE=${FULL_MIX_JUDGE_API_BASE:-http://127.0.0.1:8000/v1}
export FULL_MIX_JUDGE_API_KEY=${FULL_MIX_JUDGE_API_KEY:-EMPTY}
export FULL_MIX_JUDGE_MODEL=${FULL_MIX_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export FULL_MIX_JUDGE_TIMEOUT=${FULL_MIX_JUDGE_TIMEOUT:-120}
export FULL_MIX_JUDGE_MAX_RETRIES=${FULL_MIX_JUDGE_MAX_RETRIES:-10}
export FULL_MIX_JUDGE_MAX_TOKENS=${FULL_MIX_JUDGE_MAX_TOKENS:-32768}
export FULL_MIX_JUDGE_FORCE_JSON=${FULL_MIX_JUDGE_FORCE_JSON:-1}
export FULL_MIX_JUDGE_DISABLE_THINKING=${FULL_MIX_JUDGE_DISABLE_THINKING:-0}
export FULL_MIX_JUDGE_CONCURRENCY=${FULL_MIX_JUDGE_CONCURRENCY:-32}
export FULL_MIX_IDENTITY_JUDGE_MODEL=${FULL_MIX_IDENTITY_JUDGE_MODEL:-}


python3 -m verl.experimental.fully_async_policy.fully_async_main \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD}" \
  algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
  algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.prompt_key=prompt \
  data.truncation='error' \
  data.filter_overlong_prompts=True \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  data.gen_batch_size="${GEN_PROMPT_BSZ}" \
  data.return_raw_chat=True \
  \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}" \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_MINI_BSZ}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.actor.use_rollout_log_probs="${USE_ROLLOUT_LOG_PROBS}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OFFLOAD}" \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
  actor_rollout_ref.rollout.temperature="${TEMP}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${TEMP}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_OFFLOAD}" \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
  \
  reward.reward_manager.name=dapo \
  reward.custom_reward_function.path="${REWARD_FN_PATH}" \
  reward.custom_reward_function.name="${REWARD_FN_NAME}" \
  +reward.reward_kwargs.overlong_buffer_cfg.enable="${ENABLE_OVERLONG_BUFFER}" \
  +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}" \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR}" \
  +reward.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LEN}" \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.val_before_train=True \
  trainer.resume_mode="${RESUME_MODE}" \
  ${RESUME_FROM_PATH:++trainer.resume_from_path=${RESUME_FROM_PATH}} \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.nnodes="${NNODES_TRAIN}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.test_freq="${TEST_FREQ}" \
  \
  rollout.nnodes="${NNODES_ROLLOUT}" \
  rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
  rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
  \
  async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
  async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}" \
  async_training.require_batches="${REQUIRE_BATCHES}" \
  async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
  ${RAY_ENV_FLAGS:-} \
  "$@"
