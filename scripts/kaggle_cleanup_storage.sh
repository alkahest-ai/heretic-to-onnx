#!/usr/bin/env bash
# Delete superseded Rally Kaggle kernels and local download caches to free space.
set -euo pipefail

KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
DELETE_KERNELS="${KAGGLE_DELETE_KERNELS:-thomasjvu/rally-e4b-sft-jun14v9 thomasjvu/rally-12b-heretic-a100}"
CLEAN_TMP="${KAGGLE_CLEAN_TMP:-1}"
TMP_PATTERNS="${KAGGLE_TMP_PATTERNS:-/tmp/e4b_sft_error /tmp/12b_sft_full /tmp/12b_scorecard_v2_error /tmp/12b_scorecard_error /tmp/e4b_sft_v10_report /tmp/e2b_v3_report /tmp/e4b_compare_v3 /tmp/e4b_compare_v4_report /tmp/12b_scorecard_report}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

echo "Deleting superseded Kaggle kernels..."
for kernel in ${DELETE_KERNELS}; do
  if [[ -z "${kernel}" ]]; then
    continue
  fi
  status="$("${KAGGLE_BIN}" kernels status "${kernel}" 2>&1 | awk -F'"' '/status/{print $2; exit}')" || true
  if [[ -z "${status}" ]]; then
    echo "  skip ${kernel} (not found)"
    continue
  fi
  if [[ "${status}" == "KernelWorkerStatus.RUNNING" ]]; then
    echo "  skip ${kernel} (still RUNNING)"
    continue
  fi
  echo "  delete ${kernel} (was ${status})"
  "${KAGGLE_BIN}" kernels delete -y "${kernel}" 2>&1 || true
done

if [[ "${CLEAN_TMP}" == "1" ]]; then
  echo "Cleaning local /tmp download caches..."
  for path in ${TMP_PATTERNS}; do
    if [[ -e "${path}" ]]; then
      echo "  rm -rf ${path}"
      rm -rf "${path}"
    fi
  done
fi

echo "Cleanup done. Active Rally kernels to keep:"
for kernel in \
  thomasjvu/rally-e2b-heretic-compare \
  thomasjvu/rally-e4b-sft-jun14v10 \
  thomasjvu/rally-e4b-compare-jun14v2 \
  thomasjvu/rally-e4b-compare-jun14v10 \
  thomasjvu/rally-12b-two-stage-sft-a100 \
  thomasjvu/rally-12b-scorecard-a100; do
  status="$("${KAGGLE_BIN}" kernels status "${kernel}" 2>&1 | awk -F'"' '/status/{print $2; exit}')" || status=missing
  echo "  ${kernel}: ${status}"
done