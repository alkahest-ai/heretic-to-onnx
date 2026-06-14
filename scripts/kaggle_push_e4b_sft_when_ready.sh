#!/usr/bin/env bash
# Push E4B two-stage SFT when a Kaggle GPU slot is free.
# Copies kernel files to /tmp first — repo-path pushes can hit "Notebook not found".
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL_SRC="${KAGGLE_KERNEL_SRC:-kaggle/rally_e4b_two_stage_sft}"
PUSH_DIR="${KAGGLE_PUSH_DIR:-/tmp/rally-e4b-sft-push}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

echo "Waiting for GPU slot, then pushing ${KERNEL_SRC} via ${PUSH_DIR}..."
while true; do
  rm -rf "${PUSH_DIR}"
  mkdir -p "${PUSH_DIR}"
  cp "${ROOT_DIR}/${KERNEL_SRC}/__notebook__.ipynb" "${PUSH_DIR}/"
  cp "${ROOT_DIR}/${KERNEL_SRC}/kernel-metadata.json" "${PUSH_DIR}/"
  output="$("${KAGGLE_BIN}" kernels push \
    -p "${PUSH_DIR}" \
    --accelerator NvidiaTeslaT4 \
    --timeout 21600 2>&1)" || true
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