#!/usr/bin/env bash
# Single source of truth for environment configuration: qlcm/runtime_env.yaml.
#
# Sourced by every train_*.sh and by run_pipeline.sh. It does two jobs:
#
#   1. Loads the YAML's `env_vars:` block into THIS process, filling only
#      variables that are not already set. Precedence ends up as:
#
#           shell env  >  runtime_env.yaml  >  train-script defaults
#
#      so a one-off `QLCM_JUDGE_MODEL=x bash qlcm/train_general.sh` still wins,
#      and anything absent from the YAML falls through to the script default.
#
#   2. Builds _QLCM_RAY_FLAGS[], a bash array of Hydra overrides that push the
#      resolved values into `ray_kwargs.ray_init.runtime_env.env_vars`, so they
#      reach the Ray ACTORS — rollout workers and, critically, reward workers.
#      Exports in a shell script only ever configure the driver; actors take
#      their environment from the job runtime_env or the raylet.
#
# That second job is why the YAML alone is not enough when you run a training
# script directly: `runtime_env.yaml` is only read by `ray job submit`, and
# nothing auto-discovers it. This helper closes that gap, so the two launch
# paths behave identically:
#
#   bash qlcm/run_pipeline.sh                                    # direct
#   ray job submit --runtime-env=qlcm/runtime_env.yaml -- ...    # submitted
#
# Under `ray job submit --runtime-env=...` Ray applies the YAML itself, to the
# entrypoint AND to every actor, so BOTH steps stand down: step 1 finds each
# variable already set, and step 2 emits nothing. Step 2 standing down is not
# an optimisation — Ray rejects a driver runtime_env that repeats any key the
# job runtime_env already declares, even with an identical value, and the job
# fails at ray.init(). See _qlcm_job_env_present.
#
# Env knobs:
#   QLCM_RUNTIME_ENV   path to the YAML (default: alongside this script)
#   QLCM_NO_RAY_FLAGS  set to 1 to skip building _QLCM_RAY_FLAGS

_QLCM_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QLCM_RUNTIME_ENV="${QLCM_RUNTIME_ENV:-$_QLCM_ENV_DIR/runtime_env.yaml}"

# Variables pushed through to Ray actors. A key is propagated if it is set AND
# matches one of these prefixes, or if it appeared in the YAML's env_vars block.
_QLCM_PROPAGATE_PREFIXES=(
  QLCM_ QA_JUDGE_ OPENROUTER_API_KEY WANDB_ IF_USE_LLM_JUDGES
  INSTRUCTION_FOLLOWING_LABELS_CSV CHAT_VARIANT
  VLLM_ NCCL_ TOKENIZERS_PARALLELISM CUDA_DEVICE_MAX_CONNECTIONS
  TORCH_NCCL_ PYTORCH_ALLOC_CONF HF_ TRANSFORMERS_
)

# This helper's own knobs — they configure the loader, not the workers.
_QLCM_PROPAGATE_EXCLUDE=" QLCM_RUNTIME_ENV QLCM_NO_RAY_FLAGS QLCM_ENV_VERBOSE "

_qlcm_should_propagate() {
  local key="$1" p
  [[ "$_QLCM_PROPAGATE_EXCLUDE" == *" $key "* ]] && return 1
  for p in "${_QLCM_PROPAGATE_PREFIXES[@]}"; do
    [[ "$key" == "$p"* ]] && return 0
  done
  return 1
}

# Keys seen in the YAML, so they propagate even without a matching prefix.
_QLCM_YAML_KEYS=()

_qlcm_load_yaml() {
  [[ -f "$QLCM_RUNTIME_ENV" ]] || return 0

  local in_block=0 line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    # `env_vars:` at column 0 opens the block; any other column-0 key closes it.
    if [[ "$line" =~ ^env_vars:[[:space:]]*$ ]]; then in_block=1; continue; fi
    if [[ "$line" =~ ^[^[:space:]#] ]]; then in_block=0; continue; fi
    (( in_block )) || continue

    # Indented  KEY: value  /  KEY: "value"  /  KEY: "value"  # comment
    if [[ "$line" =~ ^[[:space:]]+([A-Za-z_][A-Za-z0-9_]*):[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      case "$val" in
        # Quoted: the value ends at the closing quote, so anything after it —
        # including a trailing comment — is not part of the value.
        \"*) val="${val#\"}"; val="${val%%\"*}" ;;
        \'*) val="${val#\'}"; val="${val%%\'*}" ;;
        # Unquoted: a '#' starts a comment.
        *)   val="${val%%#*}"; val="${val%"${val##*[![:space:]]}"}" ;;
      esac
      _QLCM_YAML_KEYS+=("$key")
      # Fill only if unset or empty — the shell environment always wins.
      if [[ -z "${!key:-}" ]]; then
        export "$key=$val"
      fi
    fi
  done < "$QLCM_RUNTIME_ENV"
}

# True when we are running under `ray job submit --runtime-env=...`.
#
# Ray puts the job's runtime_env in this variable as JSON, applies it to the
# entrypoint AND to every actor the job creates, and then REFUSES to merge a
# driver-supplied runtime_env that repeats any of the same keys — identical
# values included ("Specifying the same runtime_env fields or the same
# environment variable keys is not allowed"). So when the job already carries
# env_vars, emitting our own overrides is not redundant, it is fatal.
#
# A job submitted WITHOUT --runtime-env still sets this variable but carries no
# env_vars, and there we do need to propagate — hence the inner check.
_qlcm_job_env_present() {
  local blob="${RAY_JOB_CONFIG_JSON_ENV_VAR:-}"
  [[ -z "$blob" ]] && return 1
  case "$blob" in *'"env_vars"'*) return 0 ;; esac
  return 1
}

# Hydra override syntax treats an unquoted value containing commas as a list,
# which would turn QLCM_JUDGE_PROVIDER_ONLY into ["baidu","deepinfra",...] and
# fail Ray's "env_vars values must be strings" check. Single-quote every value
# so Hydra parses it as an opaque string. The quotes are literal characters in
# the argv entry; Hydra strips them.
_qlcm_build_ray_flags() {
  [[ "${QLCM_NO_RAY_FLAGS:-0}" == "1" ]] && return 0
  if _qlcm_job_env_present; then
    _QLCM_RAY_FLAGS=()
    if [[ "${QLCM_ENV_VERBOSE:-0}" == "1" ]]; then
      echo "[load_runtime_env] ray job runtime_env detected; not re-declaring env vars" >&2
    fi
    return 0
  fi

  local key seen=" " v
  _QLCM_RAY_FLAGS=()

  for key in "${_QLCM_YAML_KEYS[@]}" $(compgen -v | sort -u); do
    [[ "$seen" == *" $key "* ]] && continue
    seen+="$key "
    _qlcm_should_propagate "$key" || {
      # YAML keys propagate regardless of prefix.
      [[ " ${_QLCM_YAML_KEYS[*]} " == *" $key "* ]] || continue
    }
    v="${!key:-}"
    [[ -z "$v" ]] && continue
    # A literal single quote cannot be expressed inside a single-quoted Hydra
    # string; skip rather than emit something that parses as garbage.
    [[ "$v" == *"'"* ]] && { echo "[load_runtime_env] skipping $key: value contains a single quote" >&2; continue; }
    _QLCM_RAY_FLAGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.${key}='${v}'")
  done
}

_qlcm_load_yaml
_qlcm_build_ray_flags

if [[ "${QLCM_ENV_VERBOSE:-0}" == "1" ]]; then
  echo "[load_runtime_env] file: $QLCM_RUNTIME_ENV $([[ -f "$QLCM_RUNTIME_ENV" ]] || echo '(absent)')" >&2
  echo "[load_runtime_env] propagating ${#_QLCM_RAY_FLAGS[@]} vars to Ray actors" >&2
  printf '  %s\n' "${_QLCM_RAY_FLAGS[@]//=*/=<value>}" >&2
fi
