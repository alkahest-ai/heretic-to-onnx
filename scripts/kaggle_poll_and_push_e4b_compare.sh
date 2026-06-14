#!/usr/bin/env bash
# Poll E4B two-stage SFT and push the 3-model compare kernel when it completes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
TRAIN_KERNEL="${KAGGLE_TRAIN_KERNEL:-thomasjvu/rally-e4b-sft-jun14v4}"
COMPARE_PATH="${KAGGLE_COMPARE_PATH:-kaggle/rally_gemma4_e4b_compare}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT_COMPARE="${KAGGLE_TIMEOUT_COMPARE:-21600}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-300}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

echo "Polling ${TRAIN_KERNEL} every ${POLL_SECONDS}s..."
while true; do
  status="$("${KAGGLE_BIN}" kernels status "${TRAIN_KERNEL}" 2>&1 | awk -F'"' '/status/{print $2; exit}')"
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') status=${status:-unknown}"
  case "${status}" in
    KernelWorkerStatus.COMPLETE)
      echo "E4B SFT complete. Pushing compare from ${COMPARE_PATH}..."
      while true; do
        output="$("${KAGGLE_BIN}" kernels push \
          -p "${ROOT_DIR}/${COMPARE_PATH}" \
          --accelerator "${ACCELERATOR}" \
          --timeout "${TIMEOUT_COMPARE}" 2>&1)" || true
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
      ;;
    KernelWorkerStatus.ERROR)
      echo "E4B SFT errored. Not pushing compare." >&2
      exit 1
      ;;
    KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED|KernelWorkerStatus.STARTING)
      sleep "${POLL_SECONDS}"
      ;;
    *)
      echo "Unknown status: ${status}" >&2
      sleep "${POLL_SECONDS}"
      ;;
  esac
done