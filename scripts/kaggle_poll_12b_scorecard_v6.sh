#!/usr/bin/env bash
set -euo pipefail
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL="${KAGGLE_KERNEL:-thomasjvu/rally-12b-scorecard-jun15v6}"
OUT_DIR="${KAGGLE_OUT_DIR:-/tmp/12b_scorecard_v6_report}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-60}"
mkdir -p "${OUT_DIR}"
while true; do
  kernel_status="$("${KAGGLE_BIN}" kernels status "${KERNEL}" 2>&1 | awk -F'"' '/status/{print $2; exit}')" || true
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') ${KERNEL} status=${kernel_status:-unknown}"
  case "${kernel_status}" in
    KernelWorkerStatus.COMPLETE)
      "${KAGGLE_BIN}" kernels output "${KERNEL}" -p "${OUT_DIR}" --file-pattern '.*report\.json$'
      echo "Report: ${OUT_DIR}"
      exit 0 ;;
    KernelWorkerStatus.ERROR)
      "${KAGGLE_BIN}" kernels output "${KERNEL}" -p "${OUT_DIR}" 2>&1 || true
      exit 1 ;;
    *) sleep "${POLL_SECONDS}" ;;
  esac
done