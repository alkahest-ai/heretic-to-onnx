#!/usr/bin/env bash
# Push E4B RP text export when a Kaggle GPU slot is free.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL_PATH="${KAGGLE_KERNEL_PATH:-kaggle/rally_e4b_rp_text_export}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT="${KAGGLE_TIMEOUT:-21600}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

echo "Waiting for GPU slot, then pushing ${KERNEL_PATH}..."
while true; do
  output="$("${KAGGLE_BIN}" kernels push \
    -p "${ROOT_DIR}/${KERNEL_PATH}" \
    --accelerator "${ACCELERATOR}" \
    --timeout "${TIMEOUT}" 2>&1)" || true
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