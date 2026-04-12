#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PHALA_BIN:-}" ]]; then
  exec "${PHALA_BIN}" "$@"
fi

if command -v phala >/dev/null 2>&1; then
  exec phala "$@"
fi

if command -v bunx >/dev/null 2>&1; then
  exec bunx phala "$@"
fi

if command -v npx >/dev/null 2>&1; then
  exec npx phala "$@"
fi

echo "unable to find Phala CLI. Install with 'npm install -g phala' or use a system with npx/bunx." >&2
exit 1
