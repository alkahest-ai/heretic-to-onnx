#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-thomasjvu/rally-12b-rp-a100-b75-merged}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ASSISTANT_MODEL_ID="${ASSISTANT_MODEL_ID:-google/gemma-4-12B-it-assistant}"
ENABLE_MTP="${ENABLE_MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-1}"

args=(
  serve "$MODEL_ID"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --trust-remote-code
)

if [[ "$ENABLE_MTP" == "1" ]]; then
  speculative_config=$(python3 - <<PY
import json
print(json.dumps({
    "method": "mtp",
    "model": "${ASSISTANT_MODEL_ID}",
    "num_speculative_tokens": int("${NUM_SPECULATIVE_TOKENS}"),
}))
PY
)
  args+=(--speculative-config "$speculative_config")
fi

exec vllm "${args[@]}"