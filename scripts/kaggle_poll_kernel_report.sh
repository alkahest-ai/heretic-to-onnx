#!/usr/bin/env bash
# Poll a Kaggle kernel and download report.json when it completes.
set -euo pipefail

KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
KERNEL="${KAGGLE_KERNEL:?set KAGGLE_KERNEL e.g. thomasjvu/rally-12b-scorecard-a100}"
OUT_DIR="${KAGGLE_REPORT_DIR:-/tmp/kaggle-report}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-180}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

echo "Polling ${KERNEL} for report every ${POLL_SECONDS}s..."
while true; do
  status="$("${KAGGLE_BIN}" kernels status "${KERNEL}" 2>&1 | awk -F'"' '/status/{print $2; exit}')"
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') status=${status:-unknown}"
  case "${status}" in
    KernelWorkerStatus.COMPLETE)
      rm -rf "${OUT_DIR}"
      mkdir -p "${OUT_DIR}"
      "${KAGGLE_BIN}" kernels output "${KERNEL}" -p "${OUT_DIR}" --file-pattern '.*report\.json$'
      report="$(find "${OUT_DIR}" -name '*report.json' -type f | head -1)"
      if [[ -n "${report}" ]]; then
        python3 -c "
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
r = json.loads(p.read_text())
print('report=', p)
print(json.dumps(r, indent=2)[:8000])
" "${report}"
      fi
      exit 0
      ;;
    KernelWorkerStatus.ERROR)
      echo "Kernel errored." >&2
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