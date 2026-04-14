#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-4173}"

cd "$ROOT_DIR"
python3 scripts/serve_browser_chat.py --root "$ROOT_DIR" --port "$PORT"
