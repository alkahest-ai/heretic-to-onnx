#!/usr/bin/env bash
# Wait for E4B export prep (and GPU quota), then push RP text export.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
PREP_KERNEL="${KAGGLE_PREP_KERNEL:-thomasjvu/rally-e4b-export-prep}"
KERNEL_SRC="${KAGGLE_KERNEL_SRC:-kaggle/rally_e4b_rp_text_export}"
PUSH_DIR="${KAGGLE_PUSH_DIR:-/tmp/rally-e4b-rp-text-export-push}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT="${KAGGLE_TIMEOUT:-21600}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

echo "Polling ${PREP_KERNEL} until COMPLETE..."
while true; do
  kernel_status="$("${KAGGLE_BIN}" kernels status "${PREP_KERNEL}" 2>&1 | awk -F'"' '/status/{print $2; exit}')"
  echo "$(date -u '+%H:%M:%S UTC') prep_status=${kernel_status:-unknown}"
  case "${kernel_status}" in
    KernelWorkerStatus.COMPLETE) break ;;
    KernelWorkerStatus.ERROR)
      echo "Export prep errored; pushing RP text export anyway (notebook has path fallbacks)." >&2
      break
      ;;
    *) sleep "${POLL_SECONDS}" ;;
  esac
done

echo "Waiting for GPU slot, then pushing ${KERNEL_SRC} via ${PUSH_DIR}..."
while true; do
  rm -rf "${PUSH_DIR}"
  mkdir -p "${PUSH_DIR}"
  cp "${ROOT_DIR}/${KERNEL_SRC}/__notebook__.ipynb" "${PUSH_DIR}/"
  cp "${ROOT_DIR}/${KERNEL_SRC}/kernel-metadata.json" "${PUSH_DIR}/"
  output="$("${KAGGLE_BIN}" kernels push \
    -p "${PUSH_DIR}" \
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