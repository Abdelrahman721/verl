#!/usr/bin/env bash
# Fully-async GRPO training on the Medical QA STAGE-5 dataset (+retention).
# Medical-coding focus (ICD-10 + SNOMED CT × multilabel + instruction-following)
# concatenated with a 9 000-row retention pool.
#
# Self-contained experiment. Differences from stage-4:
#   - Datasets:       data/qlcm/medical_qa_stage5/train.parquet     (35 120 stage-5 coding rows)
#                     data/qlcm/medical_qa_stage5/retention.parquet (9 000 retention rows)
#                     data/qlcm/medical_qa_stage5/val.parquet       (1 337 stage-5 coding val rows;
#                                                                stratified 10 % subsample of the
#                                                                original 13 373-row val set,
#                                                                seed=5055. Full set preserved at
#                                                                val.parquet.full.bak.)
#                       Both train parquets share a unified 39-field
#                       extra_info struct and the same canonical
#                       reward_model struct so HF datasets can concat-load
#                       them in one shot.
#
#                       train.parquet pools (35 120):
#                         icd10_multilabel              (9 828)
#                         icd10_instruction_follow      (5 292)
#                         snomed_multilabel             (10 000)
#                         snomed_instruction_follow     (10 000)
#
#                       retention.parquet pools (9 000, seed=5051):
#                         general (chat/ifeval/safety)  (2 000)
#                         medical_qa UNSEEN             (3 000)
#                         bench UNSEEN (both rollouts)  (1 250)
#                         stage-4 coding UNMASTERED     (2 047)
#                         medical_qa UNSEEN top-up      (703)
#
#                       val.parquet pools (1 337, 10 % stratified subsample, seed=5055):
#                         icd10_multilabel              (491   of  4 914)
#                         icd10_instruction_follow      (106   of  1 059)
#                         snomed_multilabel             (500   of  5 000)
#                         snomed_instruction_follow     (240   of  2 400)
#
#   - Reward fn:      qlcm/rewards/compute_score_coding_mix.py
#                       Three-branch top-level dispatcher:
#                         CODING — data_source ∈ {ml, if, snomed_ml, snomed_if,
#                                                  sl, snomed_sl}
#                             -> qlcm.rewards.coding.compute_score
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
#                             -> qlcm.rewards.qa_openrouter_bench.compute_score
#                                OpenRouter judge for qa + conversation,
#                                deterministic bench scorers otherwise.
#                         GENERAL RETENTION — any other data_source
#                             -> qlcm.rewards.compute_score.compute_score
#                                Existing ifeval/chat/safety/identity dispatcher.
#                       All output reward dicts are normalised to the 29-key
#                       triple-union (15 coding + 23 qa_openrouter_bench + 2
#                       retention sentinels, deduped).
#   - Judges:         The IF scorer optionally consults DeepSeek (via OpenRouter)
#                       for the v08 / v09 / v11 / v12 rules. Gated by
#                       IF_USE_LLM_JUDGES (set to 1 here so v08/v09/v11/v12
#                       use the LLM judge; the other 13 rules are
#                       deterministic and don't touch the network).
#                       Reads OPENROUTER_API_KEY from runtime_env.yaml.
#                       The labels JSON (sct_id -> description) ships INSIDE
#                       the coding package at
#                         qlcm/rewards/coding/instruction_following/labels/all_codes_to_descriptions_reconciled.json
#                       so no INSTRUCTION_FOLLOWING_LABELS_CSV override is
#                       required. Override only if you want to swap it.
#
# Inherits cluster layout, sampling, KL-in-reward regime, batching, overlong
# buffer, PPO clip / loss, vLLM, FSDP and offload from the stage-4 runner.
# The QLCM_JUDGE_* retention judge block is DROPPED for this stage
# (no retention rows). It comes back when retention is wired in.
#
# Run INSIDE the container started by dev/dev.sh.

set -xeuo pipefail

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
# Two train parquets (coding + retention), one val parquet. verl loads them
# via HF datasets which concat-loads multiple files into a single dataset.
# Both train parquets MUST share the same Arrow schema — the build pipeline
# harmonises extra_info to a 39-field union and normalises the prompt struct
# field order so they concat cleanly. If you swap in different parquets,
# re-run the harmonisation pass.
DATA_DIR=${DATA_DIR:-/data/abdelrahman/verl/data/qlcm}
# Raw medical corpus produced upstream (shared, read-only). Everything else
# under DATA_DIR is derived from THIS model's own rollouts.
MEDICAL_QA_SRC=${MEDICAL_QA_SRC:-/data/abdelrahman/verl/data/medical_qa}
TRAIN_FILES=${TRAIN_FILES:-"[${DATA_DIR}/medical_qa_stage5/train.parquet,${DATA_DIR}/medical_qa_stage5/retention.parquet]"}
VAL_FILES=${VAL_FILES:-"[${DATA_DIR}/medical_qa_stage5/val.parquet]"}


# ---- experiment metadata ----
PROJECT_NAME=${PROJECT_NAME:-QLCM}
EXP_NAME=${EXP_NAME:-medical-qa-stage5-coding}


# ---- sequence lengths ----
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-8192}
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
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-/data/abdelrahman/verl/dumps/qlcm/medical_qa_stage5_rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-/data/abdelrahman/verl/dumps/qlcm/medical_qa_stage5_val}


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
# MUST be >= the longest possible sequence, i.e. the FULL prompt+response
# budget. rearrange_micro_batches() asserts max_token_len >= max_seq_len and
# dies mid-training otherwise. The 3/4 factor inherited from the 32B scripts
# gives 12288 against a 16384 ceiling here, so any sample longer than 12288
# tokens (a 5k prompt with a full 8k response, say) crashes the actor.
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2 ))}
GEN_TP=${GEN_TP:-1}                      # 1.7B fits on one GPU; TP=1 => one engine per rollout GPU
SP_SIZE=${SP_SIZE:-1}                    # no Ulysses SP needed at 1.7B
FSDP_SIZE=${FSDP_SIZE:-$(( NNODES_TRAIN * NGPUS_TRAIN ))}


# ---- offloading ----
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-False}
REF_OFFLOAD=${REF_OFFLOAD:-False}


# ---- vLLM rollout knobs ----
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}


# ---- reward dispatcher ----
# Stage-5 = coding only (no retention this stage). The compute_score_coding_mix
# dispatcher already supports both branches; the retention branch is just
# unused for now. When retention is added, drop the parquet sources into
# TRAIN_FILES and they'll route through qlcm.rewards.compute_score.
REWARD_FN_PATH=${REWARD_FN_PATH:-/workspace/verl/qlcm/rewards/compute_score_coding_mix.py}
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
#   * Retention (chat/safety/identity from qlcm.compute_score)
#                            → local vLLM judge OR a remote endpoint (reads QLCM_JUDGE_*)
# All keys live in runtime_env.yaml (gitignored).

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

# Retention judges (chat / safety / identity / global identity gate).
# Used for the general-retention pool in retention.parquet (local/dolci-*,
# local/safety-dpo-reference, etc.).
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
export QLCM_IDENTITY_JUDGE_MODEL=${QLCM_IDENTITY_JUDGE_MODEL:-}


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
