#!/usr/bin/env bash
# Wait for E2B compare to finish, then push E4B compare when a GPU slot is free.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
E2B_KERNEL="${KAGGLE_E2B_KERNEL:-thomasjvu/rally-e2b-heretic-compare}"
E4B_PATH="${KAGGLE_E4B_PATH:-kaggle/rally_gemma4_e4b_compare}"
E4B_KERNEL="${KAGGLE_E4B_KERNEL:-thomasjvu/rally-e4b-compare-jun14v10}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT_E4B="${KAGGLE_TIMEOUT_E4B:-21600}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

echo "Polling ${E2B_KERNEL} every ${POLL_SECONDS}s..."
while true; do
  kernel_status="$("${KAGGLE_BIN}" kernels status "${E2B_KERNEL}" 2>&1 | awk -F'"' '/status/{print $2; exit}')"
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') e2b_status=${kernel_status:-unknown}"
  case "${kernel_status}" in
    KernelWorkerStatus.COMPLETE)
      echo "E2B compare complete. Target E4B kernel ${E4B_KERNEL} from ${E4B_PATH}..."
      if "${KAGGLE_BIN}" kernels status "${E4B_KERNEL}" 2>&1 | grep -qE 'KernelWorkerStatus\.(RUNNING|COMPLETE|QUEUED|STARTING)'; then
        echo "E4B kernel already active; done."
        exit 0
      fi
      break
      ;;
    KernelWorkerStatus.ERROR)
      echo "E2B compare errored. Pushing E4B anyway..." >&2
      break
      ;;
    KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED|KernelWorkerStatus.STARTING)
      sleep "${POLL_SECONDS}"
      ;;
    *)
      echo "Unknown E2B status: ${kernel_status}" >&2
      sleep "${POLL_SECONDS}"
      ;;
  esac
done

echo "Waiting for GPU slot, then pushing ${E4B_PATH}..."
while true; do
  output="$("${KAGGLE_BIN}" kernels push \
    -p "${ROOT_DIR}/${E4B_PATH}" \
    --accelerator "${ACCELERATOR}" \
    --timeout "${TIMEOUT_E4B}" 2>&1)" || true
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