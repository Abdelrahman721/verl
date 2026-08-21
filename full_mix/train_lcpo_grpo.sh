#!/usr/bin/env bash
# LCPO budget training (arXiv 2503.04697) — SYNCHRONOUS (colocate) GRPO, 1 node.
#
# Teaches three modes at once, told apart only by the prompt suffix:
#   "Think for N tokens." + /think   reason within N tokens
#   /think                           reason freely, no target
#   /no_think                        no reasoning at all
#
# Two stages, picked with LCPO_STAGE:
#   a  LCPO-Exact  reward = task - alpha*|budget - n_think|
#   b  LCPO-Max    reward = task * clip(alpha*(budget - n_think) + delta, 0, 1)
#
# Stage b starts FROM a stage-a checkpoint (pass MODEL_PATH). Ship only stage b:
# stage a punishes coming in UNDER budget too, which fights "bigger budget,
# better results".
#
#   LCPO_STAGE=a bash full_mix/train_lcpo_grpo.sh
#   LCPO_STAGE=b MODEL_PATH=<stage-a ckpt> bash full_mix/train_lcpo_grpo.sh
#
# Forked from full_mix/train_sync_chat_ifeval.sh, keeping its execution model:
# all 8 GPUs are shared, vLLM generates, sleeps, the actor trains, weights
# reload, repeat. No overlap, no staleness, every batch on-policy. Only the
# model, the data and the reward differ.
#
# Run INSIDE the container started by dev/dev.sh:
#   bash dev/dev.sh
#   export FULL_MIX_JUDGE_API_KEY=...      # chat reward is LLM-as-judge

set -xeuo pipefail

export VLLM_USE_V1=1

_FM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_FM_DIR/secrets.env" ]]; then
  source "$_FM_DIR/secrets.env"
fi

export WANDB_API_KEY=${WANDB_API_KEY:-}

# ---- model ----
MODEL_PATH=${MODEL_PATH:-/root/verl/model}


# ---- data ----
DATA_DIR=${DATA_DIR:-/root/verl/data/lcpo_mix}
VAL_DIR=${VAL_DIR:-/root/verl/data/lcpo_val}

CHAT_VARIANT=${CHAT_VARIANT:-chat_with_baseline}
case "$CHAT_VARIANT" in
  chat)               _CHAT_FILE=chat_train.parquet ;;
  chat_with_baseline) _CHAT_FILE=chat_with_baseline_train.parquet ;;
  *) echo "CHAT_VARIANT must be 'chat' or 'chat_with_baseline'; got '$CHAT_VARIANT'" >&2; exit 2 ;;
esac

# Math IS trained on here, unlike the sibling scripts: it is the only slice
# with a real compute-accuracy slope, and a budget ladder trained without one
# has no monotonicity to learn. Built by full_mix/curriculum/build_lcpo_mix.py
# (8000 prompts x 3 rows = 24000, 60% budgeted / 20% free / 20% nothink).
TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/${_CHAT_FILE},${DATA_DIR}/ifeval_train.parquet,${DATA_DIR}/math_train.parquet]"}
# The budget ladder: 6 fixed rungs + unconstrained + zero, so the
# score-vs-budget curve and both retention checks come out of every validation.
# Built by full_mix/lcpo/build_val_ladder.py.
VAL_FILES=${VAL_FILES:-"[${VAL_DIR}/val_budget_00256.parquet,${VAL_DIR}/val_budget_00512.parquet,${VAL_DIR}/val_budget_01024.parquet,${VAL_DIR}/val_budget_02048.parquet,${VAL_DIR}/val_budget_04000.parquet,${VAL_DIR}/val_budget_06000.parquet,${VAL_DIR}/val_free.parquet,${VAL_DIR}/val_nothink.parquet]"}


# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-RL-Exps}
EXP_NAME=${EXP_NAME:-lcpo-budget-sync-4b-stage${LCPO_STAGE:-a}}


# ---- sequence lengths ----
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}


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


# ---- overlong buffer ----
# OFF for LCPO. It adds its penalty AFTER the score, so on top of stage b's
# multiplied reward it would drive the reward negative exactly where stage b
# already intends zero. With budgets capped at 6000 it would rarely fire anyway.
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-False}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}


# ---- LCPO ----
# stage a = LCPO-Exact, stage b = LCPO-Max (see the header).
LCPO_STAGE=${LCPO_STAGE:-a}
LCPO_DELTA=${LCPO_DELTA:-0.5}
# alpha is a reward-per-token exchange rate, so the right value depends on how
# much the task score varies within a group. The paper's 3e-4 is calibrated
# against a binary math reward spreading 0.30-0.50; ifeval spreads only
# 0.10-0.20 and chat 0.15-0.25, so one global alpha would let the length term
# outweigh the task term on those two by 1.5-3x.
#
# Gate G2: run ~50 stage-a steps, compare critic/lcpo/mode_budget/abs_len_err
# against each source's task-score spread, then set these so the length term is
# about 60% of that spread. The two non-math values are provisional scalings.
LCPO_ALPHA=${LCPO_ALPHA:-0.0003}
LCPO_ALPHA_MATH=${LCPO_ALPHA_MATH:-0.0003}
LCPO_ALPHA_IFEVAL=${LCPO_ALPHA_IFEVAL:-0.0001}
LCPO_ALPHA_CHAT=${LCPO_ALPHA_CHAT:-0.00015}


# ---- batching (SYNC semantics — differs from the async script) ----
# TRAIN_BATCH_SIZE prompts are generated, scored, then trained on before the
# next generation. TRAIN_MINI_BSZ splits that into optimizer steps:
#   64 / 32 = 2 optimizer steps per generation, matching the async script's
#   2 steps per parameter sync.
# 5996 prompts / 64 => ~93 steps per epoch.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
TRAIN_MINI_BSZ=${TRAIN_MINI_BSZ:-32}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
# In the sync trainer TEST_FREQ / SAVE_FREQ count TRAINING STEPS.
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-5}

RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
if [[ -n "$RESUME_FROM_PATH" ]]; then
  RESUME_MODE=${RESUME_MODE:-resume_path}
else
  RESUME_MODE=${RESUME_MODE:-disable}
fi


# ---- rollout / validation dump dirs ----
DUMP_ROOT=${DUMP_ROOT:-/root/verl/dumps/lcpo_budget_sync}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${DUMP_ROOT}/rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${DUMP_ROOT}/val}


# ---- cluster layout (colocate: ALL GPUs run both rollout and training) ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# ---- parallelism ----
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 1 ))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
GEN_TP=${GEN_TP:-1}
SP_SIZE=${SP_SIZE:-2}
FSDP_SIZE=${FSDP_SIZE:-${NGPUS_PER_NODE}}


# ---- offloading (REQUIRED in colocate) ----
# FSDP state and the vLLM engine share every GPU, so actor params + optimizer
# go to CPU while vLLM generates. Leaving these False is what makes colocate
# OOM. free_cache_engine additionally drops the KV cache during training.
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-True}
REF_OFFLOAD=${REF_OFFLOAD:-True}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}


# ---- vLLM rollout knobs ----
# Lower than the async script's 0.90: there the rollout GPUs were vLLM-only.
# Here the same GPU also has to hold gathered FSDP params and gradients.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}


# ---- reward dispatcher ----
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/full_mix/rewards/compute_score.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}


# ---- judge (the chat slice is scored by an LLM judge) ----
export FULL_MIX_JUDGE_API_BASE=${FULL_MIX_JUDGE_API_BASE:-https://openrouter.ai/api/v1}
export FULL_MIX_JUDGE_API_KEY=${FULL_MIX_JUDGE_API_KEY:-}
export FULL_MIX_JUDGE_MODEL=${FULL_MIX_JUDGE_MODEL:-deepseek/deepseek-v4-flash-0731}
export FULL_MIX_JUDGE_TIMEOUT=${FULL_MIX_JUDGE_TIMEOUT:-240}
export FULL_MIX_JUDGE_MAX_RETRIES=${FULL_MIX_JUDGE_MAX_RETRIES:-3}
# 8K. A verdict + a 25-word reason is ~150 tokens and healthy providers spend
# ~220; the cap only exists to stop a misbehaving endpoint. At the old 32768 a
# degenerate provider ran to 13k tokens before halting — paid for in full, and
# blocking a reward worker for minutes. 8K still leaves room for reasoning.
export FULL_MIX_JUDGE_MAX_TOKENS=${FULL_MIX_JUDGE_MAX_TOKENS:-16384}
export FULL_MIX_JUDGE_FORCE_JSON=${FULL_MIX_JUDGE_FORCE_JSON:-1}
export FULL_MIX_JUDGE_DISABLE_THINKING=${FULL_MIX_JUDGE_DISABLE_THINKING:-0}
# Reasoning stays ON. It is the judge's accuracy/latency dial, and turning it
# off changes the reward the policy is trained against — set explicitly rather
# than inherited from the client default so the choice is visible in the run.
export FULL_MIX_JUDGE_DISABLE_REASONING=${FULL_MIX_JUDGE_DISABLE_REASONING:-0}
# Pin the judge to a named set of OpenRouter providers. The model is served by
# 28 independently-operated deployments at quantizations from fp4 to bf16; they
# are not interchangeable. Left unpinned, 80 sampled calls hit 8 providers and
# every JSON failure came from one of them (Morph, bf16) emitting repetition
# loops of 4k-13k tokens. Pinning is what stopped that.
#
# ORDER is the priority list (ONLY is just a permission set — within it
# OpenRouter load-balances, which let the slowest member set the median
# latency: 8.0s balanced vs 1.5s ordered). The first three enforce structured
# output; digitalocean and baseten are overflow capacity for rate-limit spikes
# at CONCURRENCY=32, which is the failure this pool is sized against.
#
# novita/fp8 is in the pool but needs a client-side workaround, which
# judge_client._schema_rejected supplies: it advertises response_format (that
# flag only ever promises {"type":"json_object"}) WITHOUT structured_outputs
# (the flag that actually gates json_schema). OpenRouter therefore routes to
# it and novita hard-400s the schema — INVALID_REQUEST_BODY, not a transient,
# so untreated it would fail straight through to a neutral 0.5. call_judge now
# downgrades to json_object and retries: 4/4 recovered. Otherwise a good
# member — fp8, 100% 30m uptime, 89 tok/s.
#
# Excluded: fireworks — its status recovered (-2 -> 0, 92.9% -> 97.7% uptime)
# but every request still 429s with limit_source=upstream_provider_shared_pool,
# is_byok=false: Fireworks' shared capacity for this model is saturated, and
# even a single 16-token call fails (18/18 concurrent, 5/5 sequential). Attach
# a Fireworks key at openrouter.ai/settings/integrations to bill against your
# own limits instead, then it can go back in.
# Excluded: morph — advertises structured outputs + bf16 + status 0, yet
# produced the repetition loops above. Flags do not predict output quality.
#
# Caveat worth knowing: together does not reason (~41 completion tokens vs
# ~142 baidu / ~181 deepinfra on identical inputs) and scores ambiguous pairs
# ~0.13 higher as a result, so calls that spill across the pool are judged
# slightly differently. Set DISABLE_REASONING=1 to make them behave alike, at
# the cost of changing the reward the policy trains against.
export FULL_MIX_JUDGE_PROVIDER_ONLY=${FULL_MIX_JUDGE_PROVIDER_ONLY:-baidu,deepinfra,together,digitalocean,baseten,novita}
export FULL_MIX_JUDGE_PROVIDER_ORDER=${FULL_MIX_JUDGE_PROVIDER_ORDER:-baidu,deepinfra,together,digitalocean,baseten,novita}
# OFF, deliberately. With it on, OpenRouter routes only to providers that
# support every parameter we send, which excludes digitalocean and baseten
# (neither implements response_format) — pinned to either, the request 404s.
# Off, OpenRouter silently strips response_format for them and the judge falls
# back to following the prompt alone. Measured 2026-08-19 at 12 calls each,
# unconstrained: digitalocean 12/12 valid JSON, baseten 5/5 valid (7 rate-
# limited). The model complies without the constraint; _salvage_verdict in
# rewards/chat.py is the net underneath for when it does not.
export FULL_MIX_JUDGE_REQUIRE_PARAMETERS=${FULL_MIX_JUDGE_REQUIRE_PARAMETERS:-0}
# Non-zero so a retry can actually differ from the call that failed. At 0.0 a
# pinned single provider is deterministic: a malformed verdict would be
# regenerated byte-identically on all 4 attempts. Costs some reward variance.
export FULL_MIX_JUDGE_TEMPERATURE=${FULL_MIX_JUDGE_TEMPERATURE:-0.6}
export FULL_MIX_JUDGE_CONCURRENCY=${FULL_MIX_JUDGE_CONCURRENCY:-16}

if [[ -z "${FULL_MIX_JUDGE_API_KEY}" ]]; then
  echo "[preflight] FULL_MIX_JUDGE_API_KEY is empty; the chat slice would score a constant 0.5." >&2
  echo "[preflight] Set it in full_mix/secrets.env (see secrets.env.example) before launching." >&2
  exit 2
fi

# Global identity gate off — this mix has no identity data. Keep it off: under
# stage b the reward is already multiplicative, and this gate is a second
# multiplying judge with its own independent failure mode.
export FULL_MIX_GLOBAL_IDENTITY_CHECK_ENABLED=${FULL_MIX_GLOBAL_IDENTITY_CHECK_ENABLED:-0}


# ---- preflight ----
case "${LCPO_STAGE}" in
  a|b) ;;
  *) echo "LCPO_STAGE must be 'a' (Exact) or 'b' (Max); got '${LCPO_STAGE}'" >&2; exit 2 ;;
esac

# The checkpoint's chat_template.jinja is load-bearing and has been silently
# clobbered before: evaluation-lm/run.sh copies its own template over it and
# restores on exit, so a run killed the wrong way leaves an eval template behind.
# The one left last time pre-filled '<think>\n\n</think>' into every generation
# prompt, which would force EVERY rollout into no-think mode — budget training
# would then "succeed" at length zero and look fine on every curve.
python3 - "$MODEL_PATH" <<'PYEOF' || { echo "chat template preflight failed" >&2; exit 2; }
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
for label, suffix in (("budget", "\n\nThink for 800 tokens.\n\n/think"),
                      ("free", "\n\n/think"),
                      ("nothink", "\n\n/no_think")):
    s = tok.apply_chat_template([{"role": "user", "content": "probe" + suffix}],
                                add_generation_prompt=True, tokenize=False)
    if "</think>" in s:
        print(f"[preflight] FAIL {label}: template pre-fills a think block:\n{s[-120:]!r}", file=sys.stderr)
        sys.exit(1)
    if not s.endswith("<|im_start|>assistant\n"):
        print(f"[preflight] FAIL {label}: prompt does not end at the assistant header:\n{s[-120:]!r}", file=sys.stderr)
        sys.exit(1)
print("[preflight] chat template is open-ended for all three modes: OK")
PYEOF


python3 -m verl.trainer.main_ppo \
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
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  \
  actor_rollout_ref.hybrid_engine=True \
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
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OFFLOAD}" \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
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
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  reward.reward_manager.source=importlib \
  reward.reward_manager.module.path=/workspace/verl/full_mix/rewards/lcpo_reward_manager.py \
  reward.reward_manager.name=LCPORewardManager \
  reward.custom_reward_function.path="${REWARD_FN_PATH}" \
  reward.custom_reward_function.name="${REWARD_FN_NAME}" \
  +reward.reward_kwargs.lcpo.stage="${LCPO_STAGE}" \
  +reward.reward_kwargs.lcpo.alpha="${LCPO_ALPHA}" \
  +reward.reward_kwargs.lcpo.delta="${LCPO_DELTA}" \
  "+reward.reward_kwargs.lcpo.alpha_by_source={local/dolci-math-7b:${LCPO_ALPHA_MATH},local/dolci-ifeval-32b:${LCPO_ALPHA_IFEVAL},local/dolci-chat-32b:${LCPO_ALPHA_CHAT}}" \
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
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.test_freq="${TEST_FREQ}" \
  ${RAY_ENV_FLAGS:-} \
  "$@"
