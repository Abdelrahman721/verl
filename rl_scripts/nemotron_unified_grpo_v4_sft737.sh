set -x

# v4 reward on the unified corpus, from the Qwen3-4B-Base-sft-737 checkpoint.
# Identical to nemotron_unified_grpo_sft737.sh (and therefore nemotron_unified_grpo.sh) except
# for the reward manager and run names below; Hydra takes the last value for a key.
#
# v4 (verl/utils/reward_score/nemotron_unified_v4.py) is v3 behind a strict format gate:
#   * the response must be <think>...</think> + action, exactly one of each think tag, or it
#     gets the row's floor (-1.5 tool rows, -1 prose rows) with no judge call.
#   * prose (including malformed calls) on a tool-call row scores -1.5, below any call.
#   * tool-call markup must be whitespace-separated <tool_call>{"name", "arguments": {...}}
#     </tool_call> blocks and nothing else; any deviation is prose (format_error=1), never a
#     partially recovered call.
#
# Ray's temp dir is moved onto the shared mount: inside the container /tmp is the node's
# Docker disk, which filled up (>95%) on the first sft737 launch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${REPO_ROOT}/.ray_tmp"

EXP_NAME="grpo_nemotron_unified_v4_sft737"

exec bash "${SCRIPT_DIR}/nemotron_unified_grpo_sft737.sh" \
  reward.reward_manager.name=NemotronJudgeV4RewardManager \
  reward.reward_manager.module.path=verl/workers/reward_manager/nemotron_judge_v4.py \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.rollout_data_dir=/workspace/verl/verl_dumps/rollouts_nemotron_unified_v4_sft737 \
  trainer.validation_data_dir=/workspace/verl/verl_dumps/val_nemotron_unified_v4_sft737 \
  +ray_kwargs.ray_init._temp_dir="${REPO_ROOT}/.ray_tmp" \
  "$@"
