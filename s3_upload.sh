#!/usr/bin/env bash
#
# upload_checkpoint.sh
# -------------------------------------------------------------------
# Upload a HuggingFace-compatible model folder to S3 using the layout:
#
#     s3://<bucket>/<family>/<version>/<artifacts...>
#
# e.g.  s3://my-org-models/llama-myfork/v0.0.0/config.json
#                                              /model-00001-of-00003.safetensors
#                                              /tokenizer.json
#                                              /metadata.json
#                                              ...
#
# Key behaviours:
#   * Treats each <version> prefix as IMMUTABLE: refuses to upload if the
#     prefix already contains objects (override with --force).
#   * Validates the input looks like a HF folder (config.json present).
#   * Auto-generates a metadata.json (date, git commit, file list) if the
#     source folder doesn't already provide one — never clobbers yours.
#   * Uses `aws s3 sync`, so re-runs only transfer missing/changed files
#     and interrupted uploads resume cleanly.
#
# To add a size/variant level (e.g. .../v0.0.0/7b/...), set --variant 7b.
# -------------------------------------------------------------------

set -euo pipefail

# ----------------------------- defaults ----------------------------
BUCKET=""
FAMILY=""
VERSION=""
LOCAL_DIR=""
VARIANT=""          # optional extra prefix level, e.g. "7b"
PROFILE=""          # AWS CLI profile
REGION=""           # AWS region (only used if bucket must be addressed)
DRY_RUN="false"
FORCE="false"

# Patterns never uploaded. Edit to taste (e.g. add "optimizer.pt" to skip
# training-only state you don't want in a release).
EXCLUDES=( ".git/*" "__pycache__/*" "*.DS_Store" "*.tmp" "*.lock" )

# ----------------------------- helpers -----------------------------
err()  { printf 'ERROR: %s\n' "$*" >&2; }
info() { printf '==> %s\n' "$*"; }

usage() {
  cat <<'EOF'
Usage:
  upload_checkpoint.sh -d <local_dir> -b <bucket> -f <family> -v <version> [options]

Required:
  -d, --dir <path>        Local HuggingFace-compatible model folder
  -b, --bucket <name>     Target S3 bucket (no s3:// prefix)
  -f, --family <name>     Model family (e.g. llama-myfork)
  -v, --version <vX.Y.Z>  Release version (e.g. v0.0.0 or v1.2.0-20260614)

Options:
      --variant <name>    Extra prefix level (e.g. 7b) -> family/version/variant/
      --profile <name>    AWS CLI profile to use
      --region <name>     AWS region
      --dry-run           Show what would be uploaded, transfer nothing
      --force             Allow upload even if the version prefix already exists
  -h, --help              Show this help

Example:
  ./upload_checkpoint.sh \
      -d ./checkpoints/run-42 \
      -b my-org-models \
      -f llama-myfork \
      -v v0.0.0
EOF
}

# ----------------------------- arg parse ---------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir)      LOCAL_DIR="${2:-}"; shift 2 ;;
    -b|--bucket)   BUCKET="${2:-}";    shift 2 ;;
    -f|--family)   FAMILY="${2:-}";    shift 2 ;;
    -v|--version)  VERSION="${2:-}";   shift 2 ;;
    --variant)     VARIANT="${2:-}";   shift 2 ;;
    --profile)     PROFILE="${2:-}";   shift 2 ;;
    --region)      REGION="${2:-}";    shift 2 ;;
    --dry-run)     DRY_RUN="true";     shift   ;;
    --force)       FORCE="true";       shift   ;;
    -h|--help)     usage; exit 0 ;;
    *) err "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

# ----------------------------- validation --------------------------
missing=()
[[ -z "$LOCAL_DIR" ]] && missing+=("--dir")
[[ -z "$BUCKET"    ]] && missing+=("--bucket")
[[ -z "$FAMILY"    ]] && missing+=("--family")
[[ -z "$VERSION"   ]] && missing+=("--version")
if [[ ${#missing[@]} -gt 0 ]]; then
  err "Missing required argument(s): ${missing[*]}"
  usage
  exit 2
fi

command -v aws >/dev/null 2>&1 || { err "aws CLI not found on PATH."; exit 1; }

if [[ ! -d "$LOCAL_DIR" ]]; then
  err "Local dir does not exist or is not a directory: $LOCAL_DIR"
  exit 1
fi

# Version must look like vMAJOR.MINOR.PATCH with an optional build suffix.
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]]; then
  err "Version '$VERSION' must match vMAJOR.MINOR.PATCH (optionally -SUFFIX), e.g. v0.0.0 or v1.2.0-20260614"
  exit 1
fi

# HuggingFace sanity check: config.json is the canonical marker.
if [[ ! -f "$LOCAL_DIR/config.json" ]]; then
  err "No config.json found in $LOCAL_DIR — doesn't look like a HF model folder."
  exit 1
fi
# Warn (don't fail) if no weight shards are present.
if ! find "$LOCAL_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) | grep -q .; then
  err "Warning: no .safetensors or .bin weight files found at top level of $LOCAL_DIR."
fi

# ----------------------------- build prefix ------------------------
PREFIX="${FAMILY}/${VERSION}"
[[ -n "$VARIANT" ]] && PREFIX="${PREFIX}/${VARIANT}"
DEST="s3://${BUCKET}/${PREFIX}"

# Assemble shared AWS CLI args.
AWS_ARGS=()
[[ -n "$PROFILE" ]] && AWS_ARGS+=(--profile "$PROFILE")
[[ -n "$REGION"  ]] && AWS_ARGS+=(--region  "$REGION")

# ----------------------------- immutability check ------------------
info "Checking destination prefix is empty: ${DEST}/"
existing="$(aws s3 ls "${DEST}/" "${AWS_ARGS[@]}" 2>/dev/null || true)"
if [[ -n "$existing" ]]; then
  if [[ "$FORCE" == "true" ]]; then
    err "Prefix ${DEST}/ already exists — proceeding because --force was given."
  else
    err "Refusing to upload: ${DEST}/ already contains objects."
    err "A published version should be immutable. Bump the version, or pass --force to override."
    exit 1
  fi
fi

# ----------------------------- metadata ----------------------------
# Only generate metadata if the source folder doesn't already provide one.
TMP_META=""
cleanup() { [[ -n "$TMP_META" && -f "$TMP_META" ]] && rm -f "$TMP_META"; }
trap cleanup EXIT

if [[ ! -f "$LOCAL_DIR/metadata.json" ]]; then
  info "No metadata.json in source — generating one."
  TMP_META="$(mktemp)"

  commit="unknown"
  if command -v git >/dev/null 2>&1 && git -C "$LOCAL_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    commit="$(git -C "$LOCAL_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  fi

  # Build a JSON array of artifact paths, escaping quotes/backslashes.
  artifacts_json=""
  while IFS= read -r f; do
    esc="${f//\\/\\\\}"; esc="${esc//\"/\\\"}"
    if [[ -z "$artifacts_json" ]]; then
      artifacts_json="    \"$esc\""
    else
      artifacts_json="${artifacts_json},
    \"$esc\""
    fi
  done < <(cd "$LOCAL_DIR" && find . -type f ! -path './.git/*' | sed 's|^\./||' | sort)

  cat > "$TMP_META" <<EOF
{
  "model_family": "${FAMILY}",
  "version": "${VERSION}",
  "variant": "${VARIANT}",
  "uploaded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "uploaded_by": "$(whoami 2>/dev/null || echo unknown)",
  "source_path": "${LOCAL_DIR}",
  "git_commit": "${commit}",
  "artifacts": [
${artifacts_json}
  ]
}
EOF
else
  info "Source provides its own metadata.json — leaving it untouched."
fi

# ----------------------------- upload ------------------------------
SYNC_ARGS=("$LOCAL_DIR" "${DEST}/" "${AWS_ARGS[@]}")
for pat in "${EXCLUDES[@]}"; do
  SYNC_ARGS+=(--exclude "$pat")
done
[[ "$DRY_RUN" == "true" ]] && SYNC_ARGS+=(--dryrun)

info "Uploading ${LOCAL_DIR} -> ${DEST}/"
[[ "$DRY_RUN" == "true" ]] && info "(dry run — nothing will be transferred)"
aws s3 sync "${SYNC_ARGS[@]}"

# Upload generated metadata.json (separately, since it's outside LOCAL_DIR).
if [[ -n "$TMP_META" ]]; then
  if [[ "$DRY_RUN" == "true" ]]; then
    info "(dry run) would upload generated metadata.json -> ${DEST}/metadata.json"
  else
    info "Uploading generated metadata.json"
    aws s3 cp "$TMP_META" "${DEST}/metadata.json" "${AWS_ARGS[@]}"
  fi
fi

# ----------------------------- done --------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
  info "Dry run complete. No changes made."
else
  info "Done. Release available at: ${DEST}/"
fi