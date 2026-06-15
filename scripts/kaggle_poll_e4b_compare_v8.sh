#!/usr/bin/env bash
# Poll E4B fp16 compare v6 and download the report when complete.
set -euo pipefail

KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL="${KAGGLE_KERNEL:-thomasjvu/rally-e4b-compare-jun14v8-rpmerge}"
OUT_DIR="${KAGGLE_OUT_DIR:-/tmp/e4b_compare_v8_report}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-60}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

echo "Polling ${KERNEL} every ${POLL_SECONDS}s..."
while true; do
  kernel_status="$("${KAGGLE_BIN}" kernels status "${KERNEL}" 2>&1 | awk -F'"' '/status/{print $2; exit}')" || true
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') status=${kernel_status:-unknown}"
  case "${kernel_status}" in
    KernelWorkerStatus.COMPLETE)
      mkdir -p "${OUT_DIR}"
      "${KAGGLE_BIN}" kernels output "${KERNEL}" -p "${OUT_DIR}" --file-pattern '.*report\.json$'
      echo "Report downloaded to ${OUT_DIR}"
      exit 0
      ;;
    KernelWorkerStatus.ERROR)
      "${KAGGLE_BIN}" kernels output "${KERNEL}" -p "${OUT_DIR}" 2>&1 || true
      echo "Kernel errored; log in ${OUT_DIR}" >&2
      exit 1
      ;;
    KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED|KernelWorkerStatus.STARTING)
      sleep "${POLL_SECONDS}"
      ;;
    *)
      sleep "${POLL_SECONDS}"
      ;;
  esac
done