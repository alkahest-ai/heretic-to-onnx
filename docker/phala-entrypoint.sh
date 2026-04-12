#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    exit 1
  fi
}

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export MANIFEST_TEMPLATE="${MANIFEST_TEMPLATE:-/app/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml}"
export RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-/data/runtime-manifest.yaml}"
export WORK_DIR="${WORK_DIR:-/data/work}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/output}"
export SOURCE_MODEL_ID="${SOURCE_MODEL_ID:-p-e-w/gemma-4-E2B-it-heretic-ara}"
export BASE_MODEL_ID="${BASE_MODEL_ID:-google/gemma-4-E2B-it}"
export TARGET_REPO_ID="${TARGET_REPO_ID:-lightnolimit/gemma-4-E2B-it-heretic-ara-ONNX}"
export EXPORT_MODE="${EXPORT_MODE:-execute}"
export QUANTIZE_MODE="${QUANTIZE_MODE:-execute}"
export PYTHON_EXEC="${PYTHON_EXEC:-python3}"
export OPSET_VERSION="${OPSET_VERSION:-17}"
export BLOCK_SIZE="${BLOCK_SIZE:-32}"
export STRICT_ONNX="${STRICT_ONNX:-1}"
export UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
export HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-0}"
export HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"
export MIN_GPU_MEMORY_MB="${MIN_GPU_MEMORY_MB:-70000}"

require_env HF_TOKEN

mkdir -p "$(dirname "${RUNTIME_MANIFEST}")" "${WORK_DIR}" "${OUTPUT_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\r')"
  gpu_memory_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '\r[:space:]')"
  echo "detected gpu: ${gpu_name:-unknown} (${gpu_memory_mb:-unknown} MiB)"
  if [[ -n "${gpu_memory_mb}" ]] && [[ "${gpu_memory_mb}" =~ ^[0-9]+$ ]] && (( gpu_memory_mb < MIN_GPU_MEMORY_MB )); then
    echo "gpu memory ${gpu_memory_mb} MiB is below required minimum ${MIN_GPU_MEMORY_MB} MiB" >&2
    exit 1
  fi
else
  echo "nvidia-smi not found; continuing without gpu preflight" >&2
fi

echo "rendering runtime manifest to ${RUNTIME_MANIFEST}"
"${PYTHON_EXEC}" -m tools.heretic_to_onnx render-manifest \
  --template "${MANIFEST_TEMPLATE}" \
  --output "${RUNTIME_MANIFEST}" \
  --source-model-id "${SOURCE_MODEL_ID}" \
  --base-model-id "${BASE_MODEL_ID}" \
  --target-repo-id "${TARGET_REPO_ID}"

convert_args=(
  -m tools.heretic_to_onnx convert
  --config "${RUNTIME_MANIFEST}"
  --work-dir "${WORK_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --force
  --export-mode "${EXPORT_MODE}"
  --quantize-mode "${QUANTIZE_MODE}"
  --python-exec "${PYTHON_EXEC}"
  --opset-version "${OPSET_VERSION}"
  --block-size "${BLOCK_SIZE}"
)

if [[ "${STRICT_ONNX}" == "1" ]]; then
  convert_args+=(--strict-onnx)
fi

echo "running full convert pipeline"
"${PYTHON_EXEC}" "${convert_args[@]}"

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
  publish_args=(
    -m tools.heretic_to_onnx publish-hf
    --config "${RUNTIME_MANIFEST}"
    --package-dir "${OUTPUT_DIR}"
    --repo-id "${TARGET_REPO_ID}"
    --num-workers "${HF_UPLOAD_NUM_WORKERS}"
  )

  if [[ "${HF_UPLOAD_PRIVATE}" == "1" ]]; then
    publish_args+=(--private)
  fi

  echo "publishing package to Hugging Face repo ${TARGET_REPO_ID}"
  "${PYTHON_EXEC}" "${publish_args[@]}"
else
  echo "UPLOAD_TO_HF=0, skipping Hugging Face upload"
fi

echo "pipeline complete; package available at ${OUTPUT_DIR}"
