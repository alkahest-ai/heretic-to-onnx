#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
phala_cli="${repo_root}/scripts/phala_cli.sh"

pass() {
  echo "[ok] $1"
}

warn() {
  echo "[warn] $1"
}

fail() {
  echo "[fail] $1"
  exit 1
}

check_cmd() {
  local cmd="$1"
  local label="$2"
  if command -v "${cmd}" >/dev/null 2>&1; then
    pass "${label}: $(command -v "${cmd}")"
  else
    warn "${label}: not found"
  fi
}

echo "Phala operator doctor for ${repo_root}"

check_cmd docker "docker"
check_cmd node "node"
check_cmd npm "npm"
check_cmd bun "bun"

if "${phala_cli}" --version >/dev/null 2>&1; then
  pass "phala cli: $("${phala_cli}" --version)"
else
  fail "phala cli is not runnable"
fi

status_output="$("${phala_cli}" status 2>&1 || true)"
if grep -q "Logged in as:" <<<"${status_output}"; then
  pass "phala auth: $(grep 'Logged in as:' <<<"${status_output}" | head -n 1 | sed 's/^ *//')"
else
  warn "phala auth: not logged in, run '${phala_cli} login'"
fi

if [[ -f "${HOME}/.ssh/id_ed25519.pub" ]]; then
  pass "ssh public key: ${HOME}/.ssh/id_ed25519.pub"
elif [[ -f "${HOME}/.ssh/id_rsa.pub" ]]; then
  pass "ssh public key: ${HOME}/.ssh/id_rsa.pub"
else
  warn "ssh public key: not found; debug deployments with --dev-os will need one"
fi

for path in \
  "${repo_root}/docker/phala-compose.yml" \
  "${repo_root}/docker/phala-compose.debug.yml" \
  "${repo_root}/docker/phala.env.example" \
  "${repo_root}/scripts/build_phala_image.sh" \
  "${repo_root}/scripts/write_phala_env.sh" \
  "${repo_root}/scripts/phala_deploy.sh" \
  "${repo_root}/docs/phala-cli-operator-guide.md"
do
  if [[ -e "${path}" ]]; then
    pass "repo asset: ${path}"
  else
    fail "missing required repo asset: ${path}"
  fi
done

if [[ -f "${repo_root}/build/phala.env" ]]; then
  pass "generated env file: ${repo_root}/build/phala.env"
else
  warn "generated env file missing; run '${repo_root}/scripts/write_phala_env.sh ${repo_root}/build/phala.env <image-ref>'"
fi

echo
echo "Suggested next commands:"
echo "  ${repo_root}/scripts/build_phala_image.sh <image-ref> --push"
echo "  ${repo_root}/scripts/phala_deploy.sh --name heretic-onnx-h100 --env-file ${repo_root}/build/phala.env --instance-type <gpu-type>"
echo "  ${repo_root}/scripts/phala_deploy.sh --name heretic-onnx-debug --env-file ${repo_root}/build/phala.env --debug --ssh-pubkey ${HOME}/.ssh/id_ed25519.pub"
