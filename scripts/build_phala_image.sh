#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  /Users/area/heretic/scripts/build_phala_image.sh <image-ref> [--push]

Example:
  /Users/area/heretic/scripts/build_phala_image.sh ghcr.io/you/heretic-to-onnx:latest --push
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

image_ref="$1"
push_image="0"

if [[ $# -ge 2 ]]; then
  case "$2" in
    --push)
      push_image="1"
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

echo "building ${image_ref} from ${repo_root}"
docker build -f "${repo_root}/docker/Dockerfile.phala" -t "${image_ref}" "${repo_root}"

if [[ "${push_image}" == "1" ]]; then
  echo "pushing ${image_ref}"
  docker push "${image_ref}"
fi
