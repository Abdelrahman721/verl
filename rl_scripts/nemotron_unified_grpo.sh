set -x

# UNIFIED-CORPUS RUN (v3). Data: rl-data/nemotron_unified, balanced 33/33/33 across
# prose / single-call / parallel-call ground truths (88,041 train, 1,206 val).
#
# Reward (verl/utils/reward_score/nemotron_pivot_judge.py), all on [-1, 1]:
#   * prose rows  -> LLM judge (minimax-m3) asks ONLY whether the reply is coherent prose
#                    addressing the last user message; the gold is not shown. Pass +1,
#                    fail -1. Emitting a tool call here is -1 without a judge call.
#   * tool rows   -> emitted calls matched to ground-truth calls BY NAME with multiplicity.
#                    Matched call contributes its argument-level score in [-1, 1]; every
#                    extra call and every expected-but-unmade call contributes -1;
#                    divided by the ground-truth call count and clamped.
#
# Differences from the v2 run, all deliberate:
#   - the hard "call count must match exactly" gate is gone; under-/over-calling is now
#     graded on a slope, because half the tool rows here need 2+ calls and the gate made
#     "4 of 5 correct" score exactly what prose scored.
#   - zero-argument tools score +1 when called correctly (they scored 0.0 before; 2.3% of
#     calls in this corpus take no arguments).
#   - prose maps to +1/-1 rather than 1/0, so both halves share one scale. With
#     norm_adv_by_std_in_grpo=False that keeps GRPO advantages comparable across row types.
#
# Requires OPENROUTER_API_KEY. dev/dev.sh forwards no env vars, so export it in the
# container before running this.

# JUDGE VARIANT of nemotron_pivot_grpo.sh. Identical except for the reward:
#   * expected tool call -> unchanged rule reward (verl/utils/reward_score/nemotron_pivot.py)
#   * expected prose     -> parsable tool call scores 0.0 without a judge call; otherwise
#                           minimax/minimax-m3 grades the reply against the experts, given
#                           the last user message, as 1 or 0.
# Requires OPENROUTER_API_KEY in the environment. dev/dev.sh forwards no env vars, so
# export it inside the container before running this.


# GRPO on nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1: one decision step per row,
# reward = policy action vs expert action (verl/utils/reward_score/nemotron_pivot.py).
# Parquets from examples/data_preprocess/nemotron_pivot_preprocess.py (data_source=nemotron_pivot).
#
# Copied from rl_scripts/modified_grpo.sh; every change from that file is marked "# CHANGED".

# CHANGED: no SNOMED label CSV, no LLM-judge env, no wandb resume — this is a fresh run with a rule reward.

# CHANGED: point at a tool-calling SFT checkpoint. The prompt rows carry <think> blocks on every prior
# assistant turn, so the checkpoint's chat template must render assistant content verbatim
# (chat_template_tooling.jinja). Qwen3's official template would silently drop the earlier reasoning.
# This checkpoint's tokenizer_config.json carries that tooling template; the official Qwen3 template
# it shipped with is preserved at tokenizer_config.json.pre-tooling-template.bak.
MODEL_PATH="models/Qwen3-4B-Base-sft-3850"
TRAIN_FILES="/workspace/verl/rl-data/nemotron_unified/nemotron_unified_balanced_all_train.parquet"
VAL_FILES="/workspace/verl/rl-data/nemotron_unified/nemotron_unified_balanced_all_val.parquet"

PROJECT_NAME="RL-Exps"
EXP_NAME="grpo_nemotron_unified_v3"               # CHANGED: unified balanced corpus + v3 reward

# CHANGED: prompts are p99 7.3k / max 10.8k tokens (policy + ~17 tool schemas + history). 4096 would
# discard a third of the corpus via filter_overlong_prompts. 12288 keeps every row.
MAX_PROMPT_LEN=12288
# CHANGED: a response is one think block plus one call or one reply. Expert replies average 172 words;
# SFT traces run to ~800 words at p90. 4096 is ample and halves rollout memory vs 8192.
MAX_RESPONSE_LEN=4096

TEMP=1.0
TOP_P=1.0

N_SAMPLES_PER_PROMPT=8

TRAIN_BATCH_SIZE=32
GEN_BATCH_SIZE=96

CLIP_LOW=0.2
CLIP_HIGH=0.28

ROLLOUT_IS_LEVEL="token"
ROLLOUT_IS_THRESHOLD=2.0

# Binary reward: dynamic sampling drops groups where all 8 samples agree, which is common on the
# easy tail. NVIDIA already filtered the corpus to rows a strong model gets right 1-15 times in 16.
FILTER_METRIC="seq_reward"
MAX_NUM_GEN_BATCHES=10

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY before running the judge variant}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  \
  algorithm.norm_adv_by_std_in_grpo=False \
  \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.filter_overlong_prompts=True \
  reward.reward_manager.source=importlib \
  reward.reward_manager.name=NemotronJudgeRewardManager \
  reward.reward_manager.module.path=verl/workers/reward_manager/nemotron_judge.py \
  data.filter_overlong_prompts_workers=16 \
  data.truncation='error' \
  \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  +data.gen_batch_size="${GEN_BATCH_SIZE}" \
  +algorithm.filter_groups.enable=True \
  +algorithm.filter_groups.metric="${FILTER_METRIC}" \
  +algorithm.filter_groups.max_num_gen_batches="${MAX_NUM_GEN_BATCHES}" \
  \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  +algorithm.rollout_correction.rollout_is="${ROLLOUT_IS_LEVEL}" \
  algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
  algorithm.rollout_correction.bypass_mode=false \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.rollout_rs_threshold=null \
  +algorithm.rollout_correction.rollout_rs_threshold_lower=null \
  +algorithm.rollout_correction.rollout_token_veto_threshold=null \
  \
  actor_rollout_ref.actor.loss_agg_mode="token-mean" \
  \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=False \
  \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH}" \
  \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.temperature="${TEMP}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.total_epochs=1 \
  trainer.rollout_data_dir=/workspace/verl/verl_dumps/rollouts_nemotron_unified_v3 \
  trainer.validation_data_dir=/workspace/verl/verl_dumps/val_nemotron_unified_v3 \
  "$@"
