#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_PYTHON_BIN="${HOST_PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venvs/phala-export}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
VENV_PYTHON="$VENV_DIR/bin/python"

cd "$ROOT_DIR"

echo "[bootstrap-export] repo root: $ROOT_DIR"
echo "[bootstrap-export] venv dir: $VENV_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap-export] GPU inventory:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[bootstrap-export] creating virtualenv"
  "$HOST_PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "[bootstrap-export] upgrading pip tooling"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo "[bootstrap-export] installing PyTorch export stack"
"$VENV_PYTHON" -m pip install \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$TORCH_INDEX_URL"

echo "[bootstrap-export] installing repo package"
"$VENV_PYTHON" -m pip install -e .

echo "[bootstrap-export] installing ONNX/export dependencies"
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/docker/requirements-phala.txt"

echo "[bootstrap-export] converter preflight"
"$VENV_PYTHON" -m tools.heretic_to_onnx bootstrap

cat <<EOF
[bootstrap-export] complete
[bootstrap-export] use this interpreter for direct exports:
  PYTHON_BIN="$VENV_PYTHON" bash scripts/phala_run_rally_2b_direct.sh

[bootstrap-export] if you prefer activation:
  source "$VENV_DIR/bin/activate"
EOF
