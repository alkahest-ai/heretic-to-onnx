#!/usr/bin/env bash
# Push E4B two-stage SFT when a Kaggle GPU slot is free.
# Copies kernel files to /tmp first — repo-path pushes can hit "Notebook not found".
# Never re-push after a GPU-limit response; poll status until the run starts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL_SRC="${KAGGLE_KERNEL_SRC:-kaggle/rally_e4b_two_stage_sft}"
PUSH_DIR="${KAGGLE_PUSH_DIR:-/tmp/rally-e4b-sft-push}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

kernel_id="$(
  python3 -c "import json, pathlib; print(json.load(pathlib.Path('${ROOT_DIR}/${KERNEL_SRC}/kernel-metadata.json').open())['id'])"
)"

_poll_status() {
  "${KAGGLE_BIN}" kernels status "${kernel_id}" 2>&1 | awk -F'"' '/status/{print $2; exit}'
}

_wait_for_run() {
  echo "Polling ${kernel_id} every ${POLL_SECONDS}s..."
  while true; do
    status="$(_poll_status || true)"
    if [[ -n "${status}" ]]; then
      echo "$(date -u '+%H:%M:%S UTC') status=${status}"
      case "${status}" in
        KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED|KernelWorkerStatus.STARTING|KernelWorkerStatus.COMPLETE)
          return 0
          ;;
        KernelWorkerStatus.ERROR)
          return 1
          ;;
      esac
    else
      echo "$(date -u '+%H:%M:%S UTC') status=unknown"
    fi
    sleep "${POLL_SECONDS}"
  done
}

status="$(_poll_status || true)"
if [[ "${status}" =~ ^KernelWorkerStatus\.(RUNNING|QUEUED|STARTING|COMPLETE)$ ]]; then
  echo "Kernel already active: ${status}"
  exit 0
fi

echo "Pushing ${kernel_id} via ${PUSH_DIR}..."
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
  _wait_for_run
  exit $?
fi

echo "${output}" >&2
exit 1