#!/usr/bin/env bash
# Push 12B scorecard v4 once a Kaggle GPU slot is free.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL_SRC="${KAGGLE_KERNEL_SRC:-kaggle/rally_12b_scorecard}"
PUSH_DIR="${KAGGLE_PUSH_DIR:-/tmp/rally-12b-scorecard-push}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-60}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
BLOCKING_KERNELS="${KAGGLE_BLOCKING_KERNELS:-thomasjvu/rally-e4b-compare-jun14v2 thomasjvu/rally-12b-scorecard-a100}"

kernel_id="$(
  python3 -c "import json, pathlib; print(json.load(pathlib.Path('${ROOT_DIR}/${KERNEL_SRC}/kernel-metadata.json').open())['id'])"
)"

_kernel_status() {
  local slug="$1"
  "${KAGGLE_BIN}" kernels status "${slug}" 2>&1 | awk -F'"' '/status/{print $2; exit}' || true
}

_running_blockers() {
  local slug status count=0
  for slug in ${BLOCKING_KERNELS}; do
    status="$(_kernel_status "${slug}")"
    if [[ "${status}" == "KernelWorkerStatus.RUNNING" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

echo "Target kernel ${kernel_id}; push via ${PUSH_DIR} after blockers free a GPU slot..."
while true; do
  blockers="$(_running_blockers)"
  if [[ "${blockers}" -ge 2 ]]; then
    echo "$(date -u '+%H:%M:%S UTC') blockers_running=${blockers}; waiting ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi

  status="$(_kernel_status "${kernel_id}")"
  if [[ "${status}" =~ ^KernelWorkerStatus\.(RUNNING|QUEUED|STARTING)$ ]]; then
    echo "Kernel already active: ${status}"
    exit 0
  fi

  rm -rf "${PUSH_DIR}"
  mkdir -p "${PUSH_DIR}"
  cp "${ROOT_DIR}/${KERNEL_SRC}/__notebook__.ipynb" "${PUSH_DIR}/"
  cp "${ROOT_DIR}/${KERNEL_SRC}/kernel-metadata.json" "${PUSH_DIR}/"
  output="$("${KAGGLE_BIN}" kernels push \
    -p "${PUSH_DIR}" \
    --accelerator "${ACCELERATOR}" \
    --timeout 21600 2>&1)" || true
  echo "$(date -u '+%H:%M:%S UTC') ${output}"
  if [[ "${output}" == *"successfully pushed"* ]]; then
    exit 0
  fi
  if [[ "${output}" == *"Maximum batch GPU session count"* ]]; then
    echo "Still no GPU slot; waiting ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi
  echo "${output}" >&2
  exit 1
done