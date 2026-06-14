#!/usr/bin/env bash
# Poll E2B compare kernel and download report.json when it completes.
set -euo pipefail

KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
E2B_KERNEL="${KAGGLE_E2B_KERNEL:-thomasjvu/rally-e2b-heretic-compare}"
OUT_DIR="${KAGGLE_E2B_REPORT_DIR:-/tmp/e2b_v3_report}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  echo "kaggle CLI not found" >&2
  exit 1
fi

echo "Polling ${E2B_KERNEL} for report every ${POLL_SECONDS}s..."
while true; do
  status="$("${KAGGLE_BIN}" kernels status "${E2B_KERNEL}" 2>&1 | awk -F'"' '/status/{print $2; exit}')"
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') status=${status:-unknown}"
  case "${status}" in
    KernelWorkerStatus.COMPLETE)
      rm -rf "${OUT_DIR}"
      mkdir -p "${OUT_DIR}"
      "${KAGGLE_BIN}" kernels output "${E2B_KERNEL}" -p "${OUT_DIR}" --file-pattern '.*report\.json$'
      report="$(find "${OUT_DIR}" -name '*report.json' -type f | head -1)"
      if [[ -n "${report}" ]]; then
        python3 -c "
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
r = json.loads(p.read_text())
models = r.get('models') or {}
refusal = {k: (v.get('refusal_probe') or {}).get('false_refusal_count') for k, v in models.items()}
gate = {k: {'total': (v.get('gate') or {}).get('total_score'), 'minor': (v.get('gate') or {}).get('minor_score'), 'passed': (v.get('gate') or {}).get('passed')} for k, v in models.items()}
print('report=', p)
print('refusal_false_counts=', json.dumps(refusal, indent=2))
print('gate=', json.dumps(gate, indent=2))
print('ranking=', json.dumps(r.get('ranking'), indent=2))
" "${report}"
      else
        echo "No report.json found under ${OUT_DIR}" >&2
        exit 1
      fi
      exit 0
      ;;
    KernelWorkerStatus.ERROR)
      echo "E2B compare errored." >&2
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