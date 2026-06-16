#!/usr/bin/env bash
# Deprecated alias — use kaggle_poll_e4b_compare_v10.sh (canonical E4B compare).
set -euo pipefail
exec "$(dirname "$0")/kaggle_poll_e4b_compare_v10.sh" "$@"