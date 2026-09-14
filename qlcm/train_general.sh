#!/usr/bin/env bash
# Fully-async GRPO training for the QLCM general phase (stage `general`).
# IFEval + chat + safety + identity. No math — it was dropped from the
# production pipeline; GSM8K/MATH-500 survive as EVAL sets only.
#
# Based on verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_16_16.sh,
# adapted for:
#   - Qwen3-1.7B CPT+SFT actor + general (ifeval/chat/safety/identity) data
#   - max_response_length=16384 (vs DAPO 28k)
#   - overlong_buffer_len=1024 (vs DAPO 4096)
#   - require_batches=2 (vs DAPO 4)
#   - ppo_mini_batch_size=16 (vs DAPO 32)
#   - importance sampling explicit (use_rollout_log_probs=True)
#
# Resource model: rollout and trainer run on DISJOINT GPU pools.
# Total GPUs = NNODES_TRAIN*NGPUS_TRAIN + NNODES_ROLLOUT*NGPUS_ROLLOUT.
#
# Docs: https://verl.readthedocs.io/en/latest/advance/fully_async.html
# Run INSIDE the container started by dev/dev.sh.

set -xeuo pipefail

# vLLM async rollout engine requires the v1 path.
export VLLM_USE_V1=1

# Environment comes from qlcm/runtime_env.yaml — the single source of truth.
# The helper fills any variable not already set, and builds _QLCM_RAY_FLAGS so
# the resolved values reach the Ray ACTORS (reward + rollout workers), which do
# NOT inherit this shell. Under `ray job submit --runtime-env=...` Ray has
# already applied the YAML, so the load step is a no-op and nothing conflicts.
_QLCM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_runtime_env.sh
source "$_QLCM_DIR/load_runtime_env.sh"


# ---- model ----
MODEL_PATH=${MODEL_PATH:-/data/abdelrahman/qwen-sft/qlcm/out-qwen3-1.7b/coding-mixed-sft/final}


# ---- data ----
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/qlcm}

# Which chat parquet to use:
#   chat               -> chat_train.parquet
#   chat_with_baseline -> chat_with_baseline_train.parquet
CHAT_VARIANT=${CHAT_VARIANT:-chat_with_baseline}
case "$CHAT_VARIANT" in
  chat)               _CHAT_FILE=chat_train.parquet ;;
  chat_with_baseline) _CHAT_FILE=chat_with_baseline_train.parquet ;;
  *) echo "CHAT_VARIANT must be 'chat' or 'chat_with_baseline'; got '$CHAT_VARIANT'" >&2; exit 2 ;;
esac

TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/train/ifeval_train.parquet,${DATA_DIR}/train/${_CHAT_FILE},${DATA_DIR}/train/safety_train.parquet,${DATA_DIR}/train/identity_train.parquet]"}
VAL_FILES=${VAL_FILES:-"[${DATA_DIR}/eval/gsm8k_eval.parquet,${DATA_DIR}/eval/math500_eval.parquet,${DATA_DIR}/eval/ifeval_eval.parquet]"}


# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-QLCM}
EXP_NAME=${EXP_NAME:-general-phase}


# ---- sequence lengths ----
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
# Measured on the general stage 2026-09-02 over 800 sampled rollouts: the
# response-length distribution is bimodal, not long-tailed — p50 996 tokens,
# but ~10% run flat into the cap and never emit EOS. At 16384 those runaways
# held rollout slots for ~6 minutes each (processing_time p50 33s vs max 365s)
# and starved the trainer 59% of the step. 8192 halves the worst case; ~13% of
# responses clip and take the overlong penalty, up from ~2%, which is direct
# pressure to stop rambling.
#
# Measured on the GENERAL stage only. The medical/coding stages run the same
# actor so the non-termination behaviour should carry, but their length needs
# were not sampled — raise per stage with MAX_RESPONSE_LEN=... if they clip.
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}


# ---- sampling ----
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-0.7}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}


# ---- GRPO / KL knobs ----
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
KL_COEF=${KL_COEF:-0.0}
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
# Penalty zone is the LAST `OVERLONG_BUFFER_LEN` tokens of MAX_RESPONSE_LEN
# (i.e. inside the 16384-token cap, not added on top). vLLM still hard-truncates
# at MAX_RESPONSE_LEN. With these defaults the penalty zone is [15360, 16384].
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-True}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}


# ---- importance sampling (off-policy correction) ----
# Required for PPO/GRPO/DAPO when training is decoupled from rollout.
# Default in fully_async path; pinned True here to be explicit.
USE_ROLLOUT_LOG_PROBS=${USE_ROLLOUT_LOG_PROBS:-True}


# ---- async-training knobs ----
# staleness_threshold: fraction of stale samples allowed in a training batch.
#   0 = synchronous training; >0 = async (recommended <1). 0.1 is conservative.
# trigger_parameter_sync_step: local optimizer steps per weight sync.
# require_batches: how many ppo_mini_batches the trainer fetches per training call.
# partial_rollout: preempt long in-flight rollouts during param sync (caps stragglers).
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.25}
TRIGGER_SYNC_STEP=${TRIGGER_SYNC_STEP:-4}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}


# ---- batching ----
# Streaming async: data.train_batch_size is ignored (set to 0); gen_batch_size
# is fixed to 1; effective batch is governed by the async-training knobs above.
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-0}
GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-1}
TRAIN_MINI_BSZ=${TRAIN_MINI_BSZ:-16}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-100000}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
# Checkpoint retention. Each global_step_N is ~24 GB for this 1.7B actor
# (bf16 weights + fp32 Adam moments), and verl keeps ALL of them by default,
# so at SAVE_FREQ=20 a 2,500-step epoch would write ~3 TB. 10 keeps the last
# 200 steps of history for ~240 GB. Raise it if you need a wider window to
# pick a best checkpoint from; set to null to keep everything.
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-10}

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
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-/data/abdelrahman/verl/dumps/qlcm/general_rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-/data/abdelrahman/verl/dumps/qlcm/general_val}


# ---- cluster layout (rollout and trainer on DISJOINT GPU pools) ----
# Right-sized for a 1.7B actor on the 2 x 8-GPU cluster: each machine gives
# 2 GPUs to the trainer and 6 to rollout, so all 16 are used and rollout -
# the bottleneck at this model size - gets 12 single-GPU engines at GEN_TP=1.
# The pools must not overlap, so NNODES_TRAIN*NGPUS_TRAIN + NNODES_ROLLOUT*NGPUS_ROLLOUT
# has to fit the cluster. For a single 8-GPU box use NNODES_TRAIN=1
# NNODES_ROLLOUT=1. To reproduce the original 32B five-machine layout:
#   NNODES_TRAIN=2 NGPUS_TRAIN=8 NNODES_ROLLOUT=3 NGPUS_ROLLOUT=8
NNODES_TRAIN=${NNODES_TRAIN:-2}
NGPUS_TRAIN=${NGPUS_TRAIN:-2}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-2}
NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-6}


# ---- parallelism (dynamic-bsz + Ulysses SP + vLLM TP) ----
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 1 ))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
GEN_TP=${GEN_TP:-1}                      # 1.7B fits on one GPU; TP=1 => one engine per rollout GPU
SP_SIZE=${SP_SIZE:-1}                    # no Ulysses SP needed at 1.7B
FSDP_SIZE=${FSDP_SIZE:-$(( NNODES_TRAIN * NGPUS_TRAIN ))}


# ---- offloading ----
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-False}
REF_OFFLOAD=${REF_OFFLOAD:-False}


# ---- vLLM rollout knobs ----
# max_num_seqs caps concurrent sequences per engine. vLLM v1 defaults to 1024,
# which OOMs during warmup at our context length. 128 is plenty for streaming
# at N_SAMPLES_PER_PROMPT=16 and our queue depth.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}


# ---- reward dispatcher ----
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/qlcm/rewards/compute_score.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}


# ---- judge (chat + safety + identity rewards call the judge during training) ----
export QLCM_JUDGE_API_BASE=${QLCM_JUDGE_API_BASE:-https://openrouter.ai/api/v1}
export QLCM_JUDGE_API_KEY=${QLCM_JUDGE_API_KEY:-}
export QLCM_JUDGE_MODEL=${QLCM_JUDGE_MODEL:-deepseek/deepseek-v4-flash-0731}
export QLCM_JUDGE_TIMEOUT=${QLCM_JUDGE_TIMEOUT:-240}
export QLCM_JUDGE_MAX_RETRIES=${QLCM_JUDGE_MAX_RETRIES:-3}
# The cap exists to stop a misbehaving endpoint, not to bound normal output: a
# verdict plus a short reason is ~150 tokens and healthy providers spend ~220.
# At the old 32768 a degenerate provider ran to 13k tokens before halting —
# paid for in full, and blocking a reward worker for minutes.
export QLCM_JUDGE_MAX_TOKENS=${QLCM_JUDGE_MAX_TOKENS:-16384}
export QLCM_JUDGE_FORCE_JSON=${QLCM_JUDGE_FORCE_JSON:-1}
export QLCM_JUDGE_DISABLE_THINKING=${QLCM_JUDGE_DISABLE_THINKING:-0}
# Reasoning stays ON. It is the judge's accuracy/latency dial, and turning it
# off changes the reward the policy is trained against — set explicitly rather
# than inherited from the client default so the choice is visible in the run.
export QLCM_JUDGE_DISABLE_REASONING=${QLCM_JUDGE_DISABLE_REASONING:-0}
# Pin the judge to a named set of OpenRouter providers. The model is served by
# many independently-operated deployments at quantizations from fp4 to bf16;
# they are not interchangeable. Left unpinned, sampled calls spread across ~8
# providers and every JSON failure came from one of them (Morph, bf16) emitting
# repetition loops of 4k-13k tokens. Pinning is what stopped that.
#
# ORDER is the priority list; ONLY is just a permission set — within it
# OpenRouter load-balances, which lets the slowest member set the median
# latency (measured 8.0s balanced vs 1.5s ordered). The first three enforce
# structured output; baseten and baidu are overflow capacity for rate-limit
# spikes, which is the failure this pool is sized against. digitalocean was
# dropped 2026-09-02: measured 6/10 malformed verdicts, inventing key names
# ("false_foundation_lineage") the validator rejects. makora replaced it at
# 19/20 clean. deepinfra kept last (2/10 malformed) as last-resort overflow.
#
# novita/fp8 needs a client-side workaround that judge_client._schema_rejected
# supplies: it advertises response_format (which only ever promises
# {"type":"json_object"}) WITHOUT structured_outputs (the flag that actually
# gates json_schema). OpenRouter therefore routes to it and novita hard-400s
# the schema — INVALID_REQUEST_BODY, not a transient, so untreated it would
# fail straight through to a neutral reward. call_judge downgrades to
# json_object and retries. Otherwise a good member: fp8, high uptime.
#
# Excluded: fireworks — every request 429s with
# limit_source=upstream_provider_shared_pool, is_byok=false. Attach a Fireworks
# key at openrouter.ai/settings/integrations to bill against your own limits,
# then it can go back in.
# Excluded: morph — advertises structured outputs + bf16 + status 0, yet
# produced the repetition loops above. Flags do not predict output quality.
#
# Caveat: together does not reason (~41 completion tokens vs ~142 baidu /
# ~181 deepinfra on identical inputs) and scores ambiguous pairs ~0.13 higher
# as a result, so calls that spill across the pool are judged slightly
# differently. DISABLE_REASONING=1 makes them behave alike, at the cost of
# changing the reward the policy trains against.
export QLCM_JUDGE_PROVIDER_ONLY=${QLCM_JUDGE_PROVIDER_ONLY:-together,novita,makora,baseten,baidu,deepinfra}
export QLCM_JUDGE_PROVIDER_ORDER=${QLCM_JUDGE_PROVIDER_ORDER:-together,novita,makora,baseten,baidu,deepinfra}
# OFF, deliberately. With it on, OpenRouter routes only to providers that
# support every parameter we send, which excludes baseten (no response_format)
# — pinned to it, the request 404s.
# Off, OpenRouter silently strips response_format for them and the judge falls
# back to following the prompt alone; the model complies without the
# constraint, and _salvage_verdict in rewards/chat.py is the net underneath.
# NOTE: the judge client defaults this to 1, so it must be set explicitly.
export QLCM_JUDGE_REQUIRE_PARAMETERS=${QLCM_JUDGE_REQUIRE_PARAMETERS:-0}
# Non-zero so a retry can actually differ from the call that failed. At 0.0 a
# pinned single provider is deterministic: a malformed verdict would be
# regenerated byte-identically on every attempt. Costs some reward variance.
export QLCM_JUDGE_TEMPERATURE=${QLCM_JUDGE_TEMPERATURE:-0.6}
# The provider pool above is sized for bursts at 32. The LCPO runs settled on
# 16; drop to that if you see sustained 429s.
export QLCM_JUDGE_CONCURRENCY=${QLCM_JUDGE_CONCURRENCY:-32}


if [[ -z "${QLCM_JUDGE_API_KEY}" || "${QLCM_JUDGE_API_KEY}" == *REPLACE_ME* ]]; then
  echo "[preflight] QLCM_JUDGE_API_KEY is empty or still the template placeholder;" >&2
  echo "[preflight] chat/safety/identity rows would score a constant fallback" >&2
  echo "[preflight] instead of being judged." >&2
  echo "[preflight] Set it in qlcm/runtime_env.yaml (see runtime_env.yaml.example)." >&2
  exit 2
fi

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
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.nnodes="${NNODES_TRAIN}" \
  trainer.n_gpus_per_node="${NGPUS_TRAIN}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.test_freq="${TEST_FREQ}" \
  \
  rollout.nnodes="${NNODES_ROLLOUT}" \
  rollout.n_gpus_per_node="${NGPUS_ROLLOUT}" \
  rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
  \
  async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
  async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}" \
  async_training.require_batches="${REQUIRE_BATCHES}" \
  async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
  ${RAY_ENV_FLAGS:-} \
  ${_QLCM_RAY_FLAGS[@]+"${_QLCM_RAY_FLAGS[@]}"} \
  "$@"
