#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-4173}"

echo "Serving $ROOT_DIR at http://localhost:$PORT/browser-chat/"
cd "$ROOT_DIR"
python3 -m http.server "$PORT"
