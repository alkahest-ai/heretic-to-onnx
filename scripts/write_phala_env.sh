#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  /Users/area/heretic/scripts/write_phala_env.sh <output-path> <image-ref> [target-repo-id]

Example:
  /Users/area/heretic/scripts/write_phala_env.sh /Users/area/heretic/build/phala.env ghcr.io/you/heretic-to-onnx:latest
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

output_path="$1"
image_ref="$2"
target_repo_id="${3:-lightnolimit/gemma-4-E2B-it-heretic-ara-ONNX}"

mkdir -p "$(dirname "${output_path}")"
cat > "${output_path}" <<EOF
PHALA_IMAGE=${image_ref}
SOURCE_MODEL_ID=p-e-w/gemma-4-E2B-it-heretic-ara
BASE_MODEL_ID=google/gemma-4-E2B-it
TARGET_REPO_ID=${target_repo_id}
HF_TOKEN=replace-in-phala-encrypted-secrets
EOF

echo "wrote ${output_path}"
