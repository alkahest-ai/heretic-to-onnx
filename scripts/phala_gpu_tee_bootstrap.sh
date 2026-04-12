#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-70000}"

cd "$ROOT_DIR"

echo "[bootstrap] repo root: $ROOT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] GPU inventory:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ -n "$GPU_MEM_MIB" ]] && [[ "$GPU_MEM_MIB" -lt "$MIN_VRAM_MIB" ]]; then
    echo "[bootstrap] warning: GPU memory ${GPU_MEM_MIB}MiB is below the recommended floor of ${MIN_VRAM_MIB}MiB"
  fi
else
  echo "[bootstrap] warning: nvidia-smi not found"
fi

echo "[bootstrap] upgrading pip tooling"
$PYTHON_BIN -m pip install --upgrade pip setuptools wheel

echo "[bootstrap] installing repo package"
$PYTHON_BIN -m pip install -e .

echo "[bootstrap] installing training and ONNX dependencies"
$PYTHON_BIN -m pip install \
  "unsloth" \
  "unsloth_zoo" \
  "trl>=0.19.0" \
  "datasets>=3.1.0" \
  "accelerate>=1.0.0" \
  "safetensors>=0.4.3" \
  "sentencepiece>=0.2.0" \
  "onnx>=1.17.0" \
  "onnxscript>=0.1.0" \
  "onnxruntime-gpu>=1.20.0" \
  "onnxconverter-common>=1.14.0" \
  "huggingface_hub[cli]>=0.31.0" \
  "transformers>=4.57.0"

echo "[bootstrap] converter preflight"
$PYTHON_BIN -m tools.heretic_to_onnx bootstrap

echo "[bootstrap] complete"
