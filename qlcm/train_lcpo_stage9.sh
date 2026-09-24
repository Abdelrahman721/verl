#!/usr/bin/env bash
# QLCM stage 9 — LCPO-Max budget following on the stage-8 model. SYNC (colocate)
# GRPO on ONE node, the execution model of full_mix/train_lcpo_grpo.sh, not the
# fully-async two-node launch the other qlcm stages use.
#
# Two prompt modes, told apart only by a plain trailing sentence on the last
# user turn (the qlcm model has no /think markers; see qlcm/common/budget_prompt.py):
#   "...task...\n\nThink for a maximum of N tokens."   stay within N tokens
#   "...task..."                                        reason freely
#
# Reward: task score x clip(alpha_eff * (N - tokens) + delta, 0, 1) for budgeted
# rows (full_mix/rewards/lcpo.py, stage b, with the budget-proportional alpha
# floor and the flat runaway penalty of 2026-09-21); task score alone for free
# rows. The task score is qlcm's own per-domain scorer, reached through
# qlcm/rewards/compute_score_lcpo.py, with the linear length cost OFF: the
# budget term is the only length term now.
#
# Data: data/qlcm/medical_qa_stage9 (qlcm/curriculum/build_lcpo_stage9.py):
# ~8,000 prompts sampled from every domain the stage-8 model was trained on,
# x3 rows, 70% budgeted on a 256..4096 grid in steps of 256, 30% free; and a
# validation ladder at 256/512/1024/2048/4096 + free over held-out sets.
#
# Run INSIDE the container (bash dev/dev.sh), from /workspace/verl:
#   bash qlcm/train_lcpo_stage9.sh
# Judge keys come from qlcm/runtime_env.yaml (gitignored; copy the .example and
# fill it in, or point QLCM_RUNTIME_ENV at a file) or from the shell. Without
# `ray job submit` the YAML is read by qlcm/load_runtime_env.sh and forwarded to
# the Ray workers as Hydra overrides; the preflight below refuses to start if
# the keys are still empty.

set -xeuo pipefail

export VLLM_USE_V1=1

_QLCM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT_DIR="$(dirname "$_QLCM_DIR")"
# Repo root on PYTHONPATH so `import full_mix...` / `import qlcm...` resolve in
# EVERY Ray process. full_mix.main_ppo installs the per-source metric patch.
export PYTHONPATH="${_REPO_ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- environment: where values come from, and how they reach the workers ----
# qlcm/load_runtime_env.sh reads the env_vars block of the runtime env YAML
# (default qlcm/runtime_env.yaml; point QLCM_RUNTIME_ENV at another file) into
# this shell, filling only variables that are unset or empty, and builds
# _QLCM_RAY_FLAGS: Hydra overrides that push the resolved values into
# ray_kwargs.ray_init.runtime_env.env_vars, i.e. into the Ray ACTORS (reward
# and rollout workers). That is what `ray job submit --runtime-env` did for the
# other stages; here the flags do it, so the two launch paths agree.
#
# Precedence, exactly as for the other qlcm stages:
#       shell env  >  runtime_env.yaml  >  this script's defaults
# The two settings LCPO REQUIRES are forced here, ahead of everything, so
# neither the YAML nor the shell can turn them back on. Every other default is
# applied AFTER the YAML is loaded, and the Ray flags are rebuilt afterwards so
# the defaults reach the workers too.
#
# QLCM_LEN_PENALTY_ENABLE: the qlcm linear length cost is read at import in the
# workers; under LCPO the budget term replaces it, and compute_score_lcpo.py
# refuses to import with it on. QLCM_IDENTITY_GATE: under stage b the reward is
# already multiplicative and the gate would be a second multiplying judge.
export QLCM_LEN_PENALTY_ENABLE=0
export QLCM_IDENTITY_GATE=0

# shellcheck source=load_runtime_env.sh
source "$_QLCM_DIR/load_runtime_env.sh"

# ---- script defaults (only where neither the shell nor the YAML set a value) ----
# Sync mode scores 1,024 samples in one burst per step; judge latency is
# wall-clock here. qa_openrouter_bench defaults to 8 in flight.
export QA_JUDGE_CONCURRENCY=${QA_JUDGE_CONCURRENCY:-32}
export QLCM_JUDGE_CONCURRENCY=${QLCM_JUDGE_CONCURRENCY:-16}
# Keep the snomed_if scorer deterministic (its 4 judged rules go to a
# hardcoded DeepSeek judge when 1).
export IF_USE_LLM_JUDGES=${IF_USE_LLM_JUDGES:-0}
# Budget bins for the in-training LCPO curves (full_mix/per_source_metrics.py):
# on this grid the 256 and 512 bins are single budgets, which is what to watch.
# No propagate prefix, so it is also passed explicitly at the end of the command.
export LCPO_BUDGET_EDGES=${LCPO_BUDGET_EDGES:-256,512,1024,2048,4096}
# Judges (medical + general branches), same resolution as train_medical_qa_stage8.sh.
export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
export QA_JUDGE_OPENROUTER_API_KEY=${QA_JUDGE_OPENROUTER_API_KEY:-${OPENROUTER_API_KEY:-}}
export QA_JUDGE_MODEL=${QA_JUDGE_MODEL:-openai/gpt-5.4-mini}
export QA_JUDGE_MAX_TOKENS=${QA_JUDGE_MAX_TOKENS:-32768}
export QLCM_JUDGE_API_BASE=${QLCM_JUDGE_API_BASE:-https://openrouter.ai/api/v1}
export QLCM_JUDGE_API_KEY=${QLCM_JUDGE_API_KEY:-}
export QLCM_JUDGE_MODEL=${QLCM_JUDGE_MODEL:-deepseek/deepseek-v4-flash-0731}
export QLCM_JUDGE_TIMEOUT=${QLCM_JUDGE_TIMEOUT:-240}
export QLCM_JUDGE_MAX_RETRIES=${QLCM_JUDGE_MAX_RETRIES:-3}
export QLCM_JUDGE_MAX_TOKENS=${QLCM_JUDGE_MAX_TOKENS:-16384}
export QLCM_JUDGE_FORCE_JSON=${QLCM_JUDGE_FORCE_JSON:-1}
export QLCM_JUDGE_DISABLE_THINKING=${QLCM_JUDGE_DISABLE_THINKING:-0}
export QLCM_JUDGE_DISABLE_REASONING=${QLCM_JUDGE_DISABLE_REASONING:-0}
export QLCM_JUDGE_PROVIDER_ONLY=${QLCM_JUDGE_PROVIDER_ONLY:-together,novita,makora,baseten,baidu,deepinfra}
export QLCM_JUDGE_PROVIDER_ORDER=${QLCM_JUDGE_PROVIDER_ORDER:-together,novita,makora,baseten,baidu,deepinfra}
export QLCM_JUDGE_REQUIRE_PARAMETERS=${QLCM_JUDGE_REQUIRE_PARAMETERS:-0}
export QLCM_JUDGE_TEMPERATURE=${QLCM_JUDGE_TEMPERATURE:-0.6}
export QLCM_IDENTITY_JUDGE_MODEL=${QLCM_IDENTITY_JUDGE_MODEL:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}

# Rebuild the Ray flags now that the defaults above are in the environment.
# (The loader built them at source time; this is the same function, re-run.)
_qlcm_build_ray_flags


# ---- model ----
MODEL_PATH=${MODEL_PATH:-/data/abdelrahman/verl/checkpoints/QLCM/medical-qa-stage8-coding/global_step_160/merged_hf_model}


# ---- data ----
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/qlcm/medical_qa_stage9}
TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/train_coding.parquet,${DATA_DIR}/train_medical.parquet,${DATA_DIR}/train_general.parquet]"}
# The ladder: 5 fixed rungs + free, one metric series per rung and source.
VAL_FILES=${VAL_FILES:-"[${DATA_DIR}/val_budget_00256.parquet,${DATA_DIR}/val_budget_00512.parquet,${DATA_DIR}/val_budget_01024.parquet,${DATA_DIR}/val_budget_02048.parquet,${DATA_DIR}/val_budget_04096.parquet,${DATA_DIR}/val_free.parquet]"}


# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-QLCM}
EXP_NAME=${EXP_NAME:-medical-qa-stage9-lcpo}


# ---- sequence lengths ----
# Prompts up to 8192 as in stage 8 (the builder drops anything longer with the
# longest budget sentence attached). Responses 8192: budgets top out at 4096
# and the runaway cap should sit well above the largest budget.
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-8192}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}


# ---- sampling ----
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TOP_P=${VAL_TOP_P:-0.7}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}


# ---- GRPO / KL knobs (stage-8 regime) ----
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


# ---- overlong buffer ----
# OFF. Stage 8 used the DAPO buffer; here the LCPO runaway penalty supersedes
# it: a response cut by the cap or ending without </think> scores a flat
# -LCPO_RUNAWAY_PENALTY, which covers what the buffer was for and also the
# EOS-without-</think> case the buffer cannot see. Knobs kept so they exist.
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-False}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}


# ---- LCPO (stage b only) ----
LCPO_STAGE=b
# Multiplier at exactly n == N. 0.8 (the value chosen for the 4B stage-b2 run):
# full credit from 20% under budget at the small rungs, mild pull toward coming
# in under. Lower values re-introduce a shorter-is-better gradient across the
# whole under-budget range (0.5 makes full credit unreachable below 2048).
LCPO_DELTA=${LCPO_DELTA:-0.8}
# Reward-per-token slope; one value for all sources (stage b multiplies, so
# each source's length term is already scaled by its own task score).
LCPO_ALPHA=${LCPO_ALPHA:-0.00035}
# Budget-proportional floor on alpha: alpha_eff(N) = max(alpha, delta/(frac*N)),
# so the multiplier reaches zero no later than frac*N tokens past the budget.
# With 1.0 and the values above: zero at 2N for every budget on this grid up to
# 2048; at 4096 the fixed slack (N + 2286) is tighter and applies unchanged.
LCPO_MAX_SLACK_FRAC=${LCPO_MAX_SLACK_FRAC:-1.0}
# Flat reward for a response that never terminates (cut by MAX_RESPONSE_LEN or
# no </think>), in every mode. 0 would tie it with a finished wrong answer.
LCPO_RUNAWAY_PENALTY=${LCPO_RUNAWAY_PENALTY:-1.0}
# The budget governs think + answer ("total"). "think" leaves the answer body
# unconstrained and invites relocating the reasoning instead of compressing it.
LCPO_LENGTH_TARGET=${LCPO_LENGTH_TARGET:-total}
# A budgeted response thinking fewer than max(floor, frac*N) tokens is scored
# as no attempt, so `<think></think>` + answer is never the cheapest compliance.
LCPO_MIN_THINK_FLOOR=${LCPO_MIN_THINK_FLOOR:-32}
LCPO_MIN_THINK_FRAC=${LCPO_MIN_THINK_FRAC:-0.10}


# ---- batching (SYNC semantics) ----
# 64 prompts x 16 samples per step; 64/32 = 2 optimizer steps per generation.
# ~24,000 rows / 64 = 375 steps per epoch, 750 over the 2 epochs below.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
TRAIN_MINI_BSZ=${TRAIN_MINI_BSZ:-32}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
# TEST_FREQ / SAVE_FREQ count training steps.
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
DUMP_ROOT=${DUMP_ROOT:-/data/abdelrahman/verl/dumps/qlcm}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${DUMP_ROOT}/medical_qa_stage9_rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${DUMP_ROOT}/medical_qa_stage9_val}


# ---- cluster layout (colocate: ALL GPUs run both rollout and training) ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# ---- Ray: a private single-node instance, NOT the shared cluster ----
# The container usually has the multi-node cluster up (head .31 + worker .32 on
# port 6379). ray.init() with no address joins it and the placement group then
# lands wherever 8 GPUs are free, possibly the other box. address=local starts
# a private instance on this host on its own ports. num_cpus is a scheduling
# budget: unset on these 208-core boxes, Ray sizes its agents off os.cpu_count()
# and the runtime-env agent hangs. Set RAY_ADDRESS_MODE=auto to join instead.
RAY_ADDRESS_MODE=${RAY_ADDRESS_MODE:-local}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
_RAY_FLAGS=("+ray_kwargs.ray_init.address=${RAY_ADDRESS_MODE}")
if [[ "${RAY_ADDRESS_MODE}" == "local" ]]; then
  _RAY_FLAGS+=("ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}")
fi


# ---- parallelism (1.7B, colocate) ----
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
# Must cover the longest sequence: prompt + response.
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 1 ))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
GEN_TP=${GEN_TP:-1}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-${NGPUS_PER_NODE}}


# ---- offloading ----
# 1.7B params + Adam sharded 8-way is a few GB per GPU; offload would be pure
# overhead. Flip both to True if the vLLM engine OOMs at startup.
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-False}
REF_OFFLOAD=${REF_OFFLOAD:-False}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}


# ---- vLLM rollout knobs ----
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}


# ---- reward dispatcher ----
# compute_score_lcpo strips the validation "@rung" tag before routing through
# compute_score_coding_mix (coding F1 scorers, medical judge, general scorers)
# and stamps a numeric src_id for per-source training curves.
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/qlcm/rewards/compute_score_lcpo.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}


# ---- judge preflight ----
if [[ -z "${QLCM_JUDGE_API_KEY}" || "${QLCM_JUDGE_API_KEY}" == *REPLACE_ME* ]]; then
  echo "[preflight] QLCM_JUDGE_API_KEY is empty: chat/safety/identity rows would score a constant fallback." >&2
  echo "[preflight] Put it (and OPENROUTER_API_KEY) in qlcm/runtime_env.yaml (or QLCM_RUNTIME_ENV=<file>), or export it in the shell before launching." >&2
  exit 2
fi
if [[ -z "${QA_JUDGE_OPENROUTER_API_KEY}" ]]; then
  echo "[preflight] QA_JUDGE_OPENROUTER_API_KEY / OPENROUTER_API_KEY is empty: medical_qa / medical_conv rows cannot be judged." >&2
  exit 2
fi
# Everything the workers need must already be in _QLCM_RAY_FLAGS.
_want=(QLCM_LEN_PENALTY_ENABLE QLCM_JUDGE_API_KEY QA_JUDGE_OPENROUTER_API_KEY)
for _k in "${_want[@]}"; do
  if ! printf '%s\n' ${_QLCM_RAY_FLAGS[@]+"${_QLCM_RAY_FLAGS[@]}"} | grep -q "env_vars\.${_k}="; then
    echo "[preflight] ${_k} is not in the Ray runtime env; it is not in the YAML, not exported, and not covered by a propagate prefix." >&2
    exit 2
  fi
done


# ---- preflight ----
# Every GPU on this box must be free: colocate puts vLLM at GPU_MEM_UTIL of
# each card AND the FSDP shards on the same card. SKIP_GPU_PREFLIGHT=1 bypasses.
if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
  _busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | head -n "${NGPUS_PER_NODE}" \
            | awk -F', *' '$2 > 4096 {printf "gpu%s=%sMiB ", $1, $2}')
  if [[ -n "${_busy}" ]]; then
    echo "[preflight] GPUs already hold memory: ${_busy}" >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >&2 || true
    echo "[preflight] Free them (or SKIP_GPU_PREFLIGHT=1 to launch anyway) first." >&2
    exit 2
  fi
  echo "[preflight] all ${NGPUS_PER_NODE} GPUs are free: OK"
fi

# The chat template is load-bearing: it must not inject a system prompt the
# model never trained with, must not pre-fill a think block, and must end at
# the assistant header so the model emits <think> itself in both modes.
python3 - "$MODEL_PATH" <<'PYEOF' || { echo "chat template preflight failed" >&2; exit 2; }
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
if tok.convert_tokens_to_ids("</think>") != 151668 or tok.convert_tokens_to_ids("<think>") != 151667:
    print("[preflight] FAIL: think token ids differ from LCPORewardManager's constants", file=sys.stderr)
    sys.exit(1)
for label, suffix in (("budget", "\n\nThink for a maximum of 800 tokens."), ("free", "")):
    s = tok.apply_chat_template([{"role": "user", "content": "probe" + suffix}],
                                add_generation_prompt=True, tokenize=False)
    if "</think>" in s:
        print(f"[preflight] FAIL {label}: template pre-fills a think block:\n{s[-120:]!r}", file=sys.stderr)
        sys.exit(1)
    if "<|im_start|>system" in s:
        print(f"[preflight] FAIL {label}: template injects a system prompt the qlcm model never saw:\n{s[:160]!r}", file=sys.stderr)
        sys.exit(1)
    if not s.endswith("<|im_start|>assistant\n"):
        print(f"[preflight] FAIL {label}: prompt does not end at the assistant header:\n{s[-120:]!r}", file=sys.stderr)
        sys.exit(1)
print("[preflight] chat template is open-ended, marker-free and system-free for both modes: OK")
PYEOF


# full_mix.main_ppo = verl.trainer.main_ppo with full_mix.per_source_metrics
# installed inside the TaskRunner actor (critic/lcpo/* curves). See
# full_mix/main_ppo.py for why it is not a worker_process_setup_hook.
python3 -m full_mix.main_ppo \
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
  data.return_raw_chat=True \
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
  +reward.reward_kwargs.lcpo.length_target="${LCPO_LENGTH_TARGET}" \
  +reward.reward_kwargs.lcpo.min_think_floor="${LCPO_MIN_THINK_FLOOR}" \
  +reward.reward_kwargs.lcpo.min_think_frac="${LCPO_MIN_THINK_FRAC}" \
  +reward.reward_kwargs.lcpo.max_slack_frac="${LCPO_MAX_SLACK_FRAC}" \
  +reward.reward_kwargs.lcpo.runaway_penalty="${LCPO_RUNAWAY_PENALTY}" \
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
  "${_RAY_FLAGS[@]}" \
  ${_QLCM_RAY_FLAGS[@]+"${_QLCM_RAY_FLAGS[@]}"} \
  "+ray_kwargs.ray_init.runtime_env.env_vars.LCPO_BUDGET_EDGES='${LCPO_BUDGET_EDGES}'" \
  ${RAY_ENV_FLAGS:-} \
  "$@"
