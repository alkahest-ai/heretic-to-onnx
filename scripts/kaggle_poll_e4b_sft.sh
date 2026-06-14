#!/usr/bin/env bash
# Poll E4B SFT kernel until it starts running (no re-push).
set -euo pipefail

KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL_ID="${KAGGLE_E4B_SFT_KERNEL:-thomasjvu/rally-e4b-sft-jun14v6}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

echo "Polling ${KERNEL_ID} every ${POLL_SECONDS}s..."
while true; do
  status="$("${KAGGLE_BIN}" kernels status "${KERNEL_ID}" 2>&1 | awk -F'"' '/status/{print $2; exit}')"
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') status=${status:-unknown}"
  case "${status}" in
    KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED|KernelWorkerStatus.STARTING|KernelWorkerStatus.COMPLETE)
      exit 0
      ;;
    KernelWorkerStatus.ERROR)
      exit 1
      ;;
  esac
  sleep "${POLL_SECONDS}"
done