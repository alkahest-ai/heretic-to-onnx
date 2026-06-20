#!/usr/bin/env bash
# Re-run E4B export kernels after export-prep v2+ stages latest main.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
PREP_KERNEL="${KAGGLE_PREP_KERNEL:-thomasjvu/rally-e4b-export-prep}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT="${KAGGLE_TIMEOUT:-21600}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-90}"

_push_kernel() {
  local src="$1"
  local push_dir="$2"
  rm -rf "${push_dir}"
  mkdir -p "${push_dir}"
  cp "${ROOT_DIR}/${src}/__notebook__.ipynb" "${push_dir}/"
  cp "${ROOT_DIR}/${src}/kernel-metadata.json" "${push_dir}/"
  while true; do
    output="$("${KAGGLE_BIN}" kernels push \
      -p "${push_dir}" \
      --accelerator "${ACCELERATOR}" \
      --timeout "${TIMEOUT}" 2>&1)" || true
    if [[ "${output}" == *"successfully pushed"* ]]; then
      echo "${output}"
      return 0
    fi
    if [[ "${output}" == *"Maximum batch GPU session count"* ]]; then
      echo "$(date -u '+%H:%M:%S UTC') GPU full; retrying ${src} in ${POLL_SECONDS}s"
      sleep "${POLL_SECONDS}"
      continue
    fi
    echo "${output}" >&2
    return 1
  done
}

echo "Waiting for ${PREP_KERNEL} v2+ COMPLETE..."
while true; do
  kernel_status="$("${KAGGLE_BIN}" kernels status "${PREP_KERNEL}" 2>&1 | awk -F'"' '/status/{print $2; exit}')"
  echo "$(date -u '+%H:%M:%S UTC') prep_status=${kernel_status:-unknown}"
  [[ "${kernel_status}" == "KernelWorkerStatus.COMPLETE" ]] && break
  [[ "${kernel_status}" == "KernelWorkerStatus.ERROR" ]] && { echo "export prep failed" >&2; exit 1; }
  sleep "${POLL_SECONDS}"
done

echo "Pushing direct text export..."
_push_kernel "kaggle/rally_e4b_two_stage_export" "/tmp/rally-e4b-direct-text-export-v3"

echo "Pushing RP text export..."
_push_kernel "kaggle/rally_e4b_rp_text_export" "/tmp/rally-e4b-rp-text-export-v3"

echo "E4B export kernels re-pushed after fresh prep."