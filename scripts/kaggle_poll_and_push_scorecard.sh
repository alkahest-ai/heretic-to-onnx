#!/usr/bin/env bash
# Poll a Kaggle training kernel and push the scorecard kernel when it completes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
TRAIN_KERNEL="${KAGGLE_TRAIN_KERNEL:-thomasjvu/rally-12b-two-stage-sft-a100}"
SCORECARD_PATH="${KAGGLE_SCORECARD_PATH:-kaggle/rally_12b_scorecard}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT_SCORECARD="${KAGGLE_TIMEOUT_SCORECARD:-21600}"
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
      echo "Training complete. Pushing scorecard from ${SCORECARD_PATH}..."
      "${KAGGLE_BIN}" kernels push \
        -p "${ROOT_DIR}/${SCORECARD_PATH}" \
        --accelerator "${ACCELERATOR}" \
        --timeout "${TIMEOUT_SCORECARD}"
      echo "Scorecard pushed."
      exit 0
      ;;
    KernelWorkerStatus.ERROR)
      echo "Training kernel errored. Not pushing scorecard." >&2
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