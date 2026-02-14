#!/usr/bin/env bash
set -euo pipefail

# ===== Config (override via env or scripts/dev.env) =====
IMAGE="${IMAGE:-verlai/verl:vllm012.latest}"
NAME="${NAME:-verl-dev}"
WORKDIR_IN_CONTAINER="${WORKDIR_IN_CONTAINER:-/workspace/verl}"

# behavior toggles
PULL="${PULL:-0}"                 # 1 = docker pull image each run
RECREATE="${RECREATE:-1}"         # 1 = delete + recreate container each run
AS_USER="${AS_USER:-0}"           # 1 = run container as host UID/GID
MOUNT_CACHES="${MOUNT_CACHES:-1}" # 1 = mount HF/torch/vllm caches
NET_HOST="${NET_HOST:-1}"         # 1 = --net=host (recommended for vLLM/NCCL)
SHM_SIZE="${SHM_SIZE:-10g}"

# Load optional env file (lets teammates customize without editing script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/dev.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

# Repo root = parent of scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ===== Helpers =====
exists_container() { docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; }
running_container() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }

# ===== Optional: recreate =====
if [[ "$RECREATE" == "1" ]] && exists_container; then
  echo "[+] Removing container $NAME"
  docker rm -f "$NAME" >/dev/null || true
fi

# ===== Pull image =====
if [[ "$PULL" == "1" ]]; then
  echo "[+] Pulling $IMAGE"
  docker pull "$IMAGE"
fi

# ===== Create container if missing =====
if ! exists_container; then
  echo "[+] Creating container $NAME"

  # Base args
  ARGS=(
    docker create
    --runtime=nvidia --gpus all
    --shm-size="$SHM_SIZE"
    --cap-add=SYS_ADMIN
    -v "$REPO_ROOT":"$WORKDIR_IN_CONTAINER"
    -w "$WORKDIR_IN_CONTAINER"
    --name "$NAME"
  )

  # Host networking (vLLM server, NCCL, distributed, etc.)
  if [[ "$NET_HOST" == "1" ]]; then
    ARGS+=(--net=host)
  fi

  # Run as host user to avoid root-owned files on the mounted repo
  if [[ "$AS_USER" == "1" ]]; then
    ARGS+=(--user "$(id -u)":"$(id -g)")
    # Make user/group visible in container for nicer prompts/tools
    ARGS+=(-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro)
  fi

  # Cache mounts for speed (HF downloads, torch extensions, vLLM)
  if [[ "$MOUNT_CACHES" == "1" ]]; then
    mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/torch" "$HOME/.cache/vllm"

    if [[ "$AS_USER" == "1" ]]; then
      ARGS+=(
        -v "$HOME/.cache/huggingface":"/home/$(id -un)/.cache/huggingface"
        -v "$HOME/.cache/torch":"/home/$(id -un)/.cache/torch"
        -v "$HOME/.cache/vllm":"/home/$(id -un)/.cache/vllm"
      )
    else
      ARGS+=(
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface
        -v "$HOME/.cache/torch":/root/.cache/torch
        -v "$HOME/.cache/vllm":/root/.cache/vllm
      )
    fi
  fi

  # Keep it alive
  ARGS+=("$IMAGE" sleep infinity)

  "${ARGS[@]}" >/dev/null
fi

# ===== Start if not running =====
if ! running_container; then
  echo "[+] Starting container $NAME"
  docker start "$NAME" >/dev/null
fi

echo "[+] Running: pip3 install --no-deps -e ."
docker exec "$NAME" bash -lc "cd '$WORKDIR_IN_CONTAINER' && pip3 install --no-deps -e ."

echo "[+] Entering $NAME at $WORKDIR_IN_CONTAINER"
exec docker exec -it "$NAME" bash
