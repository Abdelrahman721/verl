set -x

# v3 reward on the unified corpus, from the Qwen3-4B-Base-sft-737 checkpoint.
# Identical to nemotron_unified_grpo.sh except for the overrides below; Hydra takes the
# last value for a key, so these win over the defaults baked into that script.
#
# The checkpoint was SFT'd with chat_template_tooling.jinja; its tokenizer_config.json now
# carries that template (original kept at tokenizer_config.json.pre-tooling-template.bak).
# The template it shipped with drops tool-role messages and tool_calls, so every prompt
# would have lost its <tool_call>/<tool_response> history.
#
# Keys: dev/secrets.env (gitignored) is sourced if present; OPENROUTER_API_KEY is required,
# WANDB_API_KEY is needed for the wandb logger.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# set +x around the source so the keys are not echoed into the training log.
{ set +x; } 2>/dev/null
[[ -f "${SCRIPT_DIR}/../dev/secrets.env" ]] && source "${SCRIPT_DIR}/../dev/secrets.env"
set -x

EXP_NAME="grpo_nemotron_unified_v3_sft737"

exec bash "${SCRIPT_DIR}/nemotron_unified_grpo.sh" \
  actor_rollout_ref.model.path=models/Qwen3-4B-Base-sft-737 \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.rollout_data_dir=/workspace/verl/verl_dumps/rollouts_nemotron_unified_v3_sft737 \
  trainer.validation_data_dir=/workspace/verl/verl_dumps/val_nemotron_unified_v3_sft737 \
  trainer.max_actor_ckpt_to_keep=2 \
  "$@"
