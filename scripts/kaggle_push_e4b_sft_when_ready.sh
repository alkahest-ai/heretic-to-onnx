#!/usr/bin/env bash
# Push E4B two-stage SFT when a Kaggle GPU slot is free.
# Copies kernel files to /tmp first — repo-path pushes can hit "Notebook not found".
# On GPU-limit responses, poll kernel status instead of re-pushing (re-push poisons slugs).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL_SRC="${KAGGLE_KERNEL_SRC:-kaggle/rally_e4b_two_stage_sft}"
PUSH_DIR="${KAGGLE_PUSH_DIR:-/tmp/rally-e4b-sft-push}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

kernel_id="$(
  python3 -c "import json, pathlib; print(json.load(pathlib.Path('${ROOT_DIR}/${KERNEL_SRC}/kernel-metadata.json').open())['id'])"
)"

echo "Target kernel ${kernel_id}; push via ${PUSH_DIR} when a GPU slot is free..."
while true; do
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
    echo "GPU slots full; polling ${kernel_id} (no re-push)..."
    while true; do
      status="$("${KAGGLE_BIN}" kernels status "${kernel_id}" 2>&1 | awk -F'"' '/status/{print $2; exit}')"
      if [[ -n "${status}" ]]; then
        echo "$(date -u '+%H:%M:%S UTC') status=${status}"
        case "${status}" in
          KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED|KernelWorkerStatus.STARTING|KernelWorkerStatus.COMPLETE)
            exit 0
            ;;
        esac
      else
        echo "$(date -u '+%H:%M:%S UTC') status=unknown (kernel may still be registering)"
      fi
      sleep "${POLL_SECONDS}"
    done
  fi
  echo "${output}" >&2
  exit 1
done