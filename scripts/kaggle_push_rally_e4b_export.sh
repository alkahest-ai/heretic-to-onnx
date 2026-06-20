#!/usr/bin/env bash
# Push the Rally E4B browser export lane (prep → direct text → merged upload → RP text).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT="${KAGGLE_TIMEOUT:-21600}"

kernels=(
  "kaggle/rally_e4b_export_prep"
  "kaggle/rally_e4b_two_stage_export"
  "kaggle/rally_e4b_rp_merged_upload"
  "kaggle/rally_e4b_rp_text_export"
)

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

for path in "${kernels[@]}"; do
  echo "Pushing ${path}..."
  args=(-p "${ROOT_DIR}/${path}" --timeout "${TIMEOUT}")
  if [[ "${path}" != *export_prep* ]]; then
    args+=(--accelerator "${ACCELERATOR}")
  fi
  "${KAGGLE_BIN}" kernels push "${args[@]}"
done

echo "E4B export lane pushed. Poll outputs before browser smoke."