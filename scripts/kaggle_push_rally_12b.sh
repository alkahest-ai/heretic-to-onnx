#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT_HERETIC="${KAGGLE_TIMEOUT_HERETIC:-43200}"
TIMEOUT_SFT="${KAGGLE_TIMEOUT_SFT:-43200}"
TIMEOUT_SCORECARD="${KAGGLE_TIMEOUT_SCORECARD:-21600}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: python3 -m pip install kaggle" >&2
  exit 1
fi

push_kernel() {
  local path="$1"
  local timeout="${2:-}"
  local -a cmd=("${KAGGLE_BIN}" kernels push -p "${ROOT_DIR}/${path}" --accelerator "${ACCELERATOR}")
  if [[ -n "${timeout}" ]]; then
    cmd+=(--timeout "${timeout}")
  fi
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

echo "[1/4] Heretic ablation"
push_kernel "kaggle/rally_12b_heretic" "${TIMEOUT_HERETIC}"

echo "[2/4] Two-stage RP SFT"
push_kernel "kaggle/rally_12b_two_stage_sft" "${TIMEOUT_SFT}"

echo "[3/4] RP scorecard"
push_kernel "kaggle/rally_12b_scorecard" "${TIMEOUT_SCORECARD}"

echo "[4/4] RP merged upload (CPU)"
"${KAGGLE_BIN}" kernels push -p "${ROOT_DIR}/kaggle/rally_12b_rp_merged_upload"

echo "Done. Track runs with:"
echo "  ${KAGGLE_BIN} kernels status thomasjvu/rally-12b-heretic-a100"
echo "  ${KAGGLE_BIN} kernels status thomasjvu/rally-12b-two-stage-sft-a100"
echo "  ${KAGGLE_BIN} kernels status thomasjvu/rally-12b-scorecard-a100"
echo "  ${KAGGLE_BIN} kernels status thomasjvu/rally-12b-rp-merged-upload-a100"