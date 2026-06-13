#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-thomasjvu/rally-12b-rp-a100-b75-merged}"
ASSISTANT_MODEL_ID="${ASSISTANT_MODEL_ID:-google/gemma-4-12B-it-assistant}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_MTP="${ENABLE_MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-1}"

export MODEL_ID ASSISTANT_MODEL_ID HOST PORT MAX_MODEL_LEN GPU_MEMORY_UTILIZATION ENABLE_MTP NUM_SPECULATIVE_TOKENS

if command -v vllm >/dev/null 2>&1; then
  exec bash "${ROOT_DIR}/docker/vllm-entrypoint.sh"
fi

if command -v docker >/dev/null 2>&1; then
  exec docker run --rm -it \
    --gpus all \
    --shm-size 16g \
    -p "${PORT}:${PORT}" \
    -e MODEL_ID \
    -e ASSISTANT_MODEL_ID \
    -e HOST=0.0.0.0 \
    -e PORT \
    -e MAX_MODEL_LEN \
    -e GPU_MEMORY_UTILIZATION \
    -e ENABLE_MTP \
    -e NUM_SPECULATIVE_TOKENS \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    heretic-vllm-gemma4:latest
fi

echo "Install vllm or docker to serve ${MODEL_ID}" >&2
exit 1