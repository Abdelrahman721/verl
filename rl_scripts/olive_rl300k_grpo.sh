set -x

# GRPO on the olive_rl300k corpus (rl-data/olive_rl300k), from the olive SFT checkpoint
# Qwen3-4B-Base-olive-sft-1074 (epoch 3 of 3).
#
# Data: examples/data_preprocess/olive_rl300k_preprocess.py, data_source="olive_rl300k".
#   296,814 train / 2,970 val rows, split by conversation, 7 sources.
#   Ground truths: 54% single call, 31% prose, 15% parallel batch (NOT balanced like the
#   nemotron_unified corpus; it is the natural mix of the sources).
#
# Reward: verl/utils/reward_score/olive_rl300k.py, routed by data_source inside
# NemotronJudgeRewardManager (no separate manager needed):
#   * prose rows -> LLM judge, +1 pass / -1 fail; a parsable tool call here is -1.
#   * tool rows  -> mean over EXPECTED calls of {1.0 name+keys+values, 0.5 name+keys,
#                   0.0 otherwise}, scaled by matched/emitted when surplus calls are made;
#                   prose or no parsable call is -1.
#
# Everything not set here comes from nemotron_unified_grpo.sh -> nemotron_unified_grpo_sft737.sh.
# Hydra takes the last value for a key, so these win.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${REPO_ROOT}/.ray_tmp"

# set +x around the source so the keys are not echoed into the training log.
{ set +x; } 2>/dev/null
[[ -f "${REPO_ROOT}/dev/secrets.env" ]] && source "${REPO_ROOT}/dev/secrets.env"
set -x

EXP_NAME="grpo_olive_rl300k_olivesft1074"

DATA_DIR="/workspace/verl/rl-data/olive_rl300k"
TRAIN_FILES="${DATA_DIR}/olive_rl300k_all_train.parquet"
VAL_FILES="${DATA_DIR}/olive_rl300k_all_val.parquet"

# Epoch 3 (final, step 1074) of abdelrahman-qwen-olive-32k-bs8-3ep-30gpu, copied in from
# models-sft/. Its tokenizer_config.json shipped a stripped template that renders neither
# tool-role messages nor tool_calls - it changed 220 of 300 sampled prompts - so
# chat_template_tooling.jinja was installed over it (original at
# tokenizer_config.json.pre-tooling-template.bak). It now renders byte-identically to sft-737.
MODEL_PATH="models/Qwen3-4B-Base-olive-sft-1074"

# Prompt length. Measured on 4,957 rows sampled across every row group, this checkpoint's
# tokenizer + chat_template_tooling.jinja:  p50 4,273  p90 22,283  p99 34,904  max 51,064.
# filter_overlong_prompts=True DROPS anything longer, and the long rows are not spread evenly:
#     12288 -> 23.5% dropped (93.8% of openresearcher, 87.2% of openseeker, 60.9% of
#              swe-zero-openhands): what survives is roughly the old nemotron_unified corpus.
#     16384 -> 17.2% dropped, and still 82.8% / 66.0% / 40.8% of those three sources.
#     28672 -> 3.9% dropped (23.3% / 27.7% / 7.6%), and every source is still represented.
# This checkpoint's max_position_embeddings is 32768, NOT the 40960 of Qwen3-4B-Base-sft-737
# (it was SFT'd at 32k), so prompt + response must fit 32768: 28672 + 4096 lands exactly on it.
MAX_PROMPT_LEN=28672
MAX_RESPONSE_LEN=4096

# At 32,768-token sequences a fixed micro batch of 2 is ~66k tokens per backward, and the
# logits alone (vocab 151,936) would be ~20 GB. Switch the actor and the rollout log-prob pass
# to token-budgeted batching instead; the budget must be >= one full sequence.
MAX_TOKEN_LEN_PER_GPU=32768

# 296,814 rows at train_batch_size=32 is 9,275 steps for a single epoch. Cap it explicitly so
# the run ends on a step count rather than whenever the corpus runs out.
TOTAL_TRAINING_STEPS=1000

# KEEP EVERY CHECKPOINT. nemotron_unified_grpo_sft737.sh sets max_actor_ckpt_to_keep=2, which
# prunes irreversibly as it saves - the v4 run left seven 12 KB stubs where the weights had
# been, and only its last two steps were ever convertible. null is verl's own default and
# disables pruning entirely (checkpoint_manager.py returns early on a falsy value).
# Cost at save_freq=50: ~47 GB per checkpoint, so ~940 GB to reach step 1000. /mnt/data01 has
# 451 T free, so retention is not the constraint; losing a checkpoint you wanted is.
MAX_ACTOR_CKPT_TO_KEEP=null

exec bash "${SCRIPT_DIR}/nemotron_unified_grpo_sft737.sh" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.rollout_data_dir=/workspace/verl/verl_dumps/rollouts_olive_rl300k_olivesft1074 \
  trainer.validation_data_dir=/workspace/verl/verl_dumps/val_olive_rl300k_olivesft1074 \
  +ray_kwargs.ray_init._temp_dir="${REPO_ROOT}/.ray_tmp" \
  "$@"
