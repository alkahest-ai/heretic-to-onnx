#!/usr/bin/env bash
# Push E4B two-stage SFT when a Kaggle GPU slot is free.
# Copies kernel files to /tmp first — repo-path pushes can hit "Notebook not found".
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL_SRC="${KAGGLE_KERNEL_SRC:-kaggle/rally_e4b_two_stage_sft}"
PUSH_DIR="${KAGGLE_PUSH_DIR:-/tmp/rally-e4b-sft-push}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-300}"

kernel_id="$(
  python3 -c "import json, pathlib; print(json.load(pathlib.Path('${ROOT_DIR}/${KERNEL_SRC}/kernel-metadata.json').open())['id'])"
)"

echo "Waiting for GPU slot, then pushing ${kernel_id} via ${PUSH_DIR}..."
while true; do
  status="$("${KAGGLE_BIN}" kernels status "${kernel_id}" 2>&1 | awk -F'"' '/status/{print $2; exit}')" || true
  if [[ "${status}" =~ ^KernelWorkerStatus\.(RUNNING|QUEUED|STARTING|COMPLETE)$ ]]; then
    echo "Kernel already active: ${status}"
    exit 0
  fi

  rm -rf "${PUSH_DIR}"
  mkdir -p "${PUSH_DIR}"
  cp "${ROOT_DIR}/${KERNEL_SRC}/__notebook__.ipynb" "${PUSH_DIR}/"
  cp "${ROOT_DIR}/${KERNEL_SRC}/kernel-metadata.json" "${PUSH_DIR}/"
  output="$("${KAGGLE_BIN}" kernels push \
    -p "${PUSH_DIR}" \
    --accelerator NvidiaTeslaT4 \
    --timeout 21600 2>&1)" || true
  echo "$(date -u '+%H:%M:%S UTC') ${output}"
  if [[ "${output}" == *"successfully pushed"* ]]; then
    exit 0
  fi
  if [[ "${output}" == *"Maximum batch GPU session count"* ]]; then
    echo "GPU slots full; retrying push in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi
  echo "${output}" >&2
  exit 1
done