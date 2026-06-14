#!/usr/bin/env bash
# Push E4B compare when a Kaggle GPU slot is free.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL_PATH="${KAGGLE_KERNEL_PATH:-kaggle/rally_gemma4_e4b_compare}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

echo "Waiting for GPU slot, then pushing ${KERNEL_PATH}..."
while true; do
  output="$("${KAGGLE_BIN}" kernels push -p "${ROOT_DIR}/${KERNEL_PATH}" --accelerator NvidiaTeslaT4 --timeout 21600 2>&1)" || true
  if [[ "${output}" == *"successfully pushed"* ]]; then
    echo "${output}"
    exit 0
  fi
  if [[ "${output}" == *"Maximum batch GPU session count"* ]]; then
    echo "$(date -u '+%H:%M:%S UTC') GPU slots full; retrying in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi
  echo "${output}" >&2
  exit 1
done