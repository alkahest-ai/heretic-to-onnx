#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  /Users/area/heretic/scripts/phala_deploy.sh --name <cvm-name> --env-file <env-file> [options]

Options:
  --name <value>              CVM name
  --env-file <path>           Env file passed to `phala deploy -e`
  --instance-type <value>     GPU/CPU instance type
  --region <value>            Preferred region
  --compose <path>            Override compose file
  --ssh-pubkey <path>         SSH public key path for debug/dev-os deploys
  --debug                     Use the debug compose and enable --dev-os
  --link                      Link the current repo to the deployed CVM (default)
  --no-link                   Skip `phala link`
  --wait                      Wait for deploy completion (default)
  --no-wait                   Do not wait for deploy completion
  --public-logs               Enable public logs
  --public-sysinfo            Enable public sysinfo
  --listed                    List CVM publicly
  --help                      Show this help

Examples:
  /Users/area/heretic/scripts/phala_deploy.sh \
    --name heretic-onnx-h100 \
    --env-file /Users/area/heretic/build/phala.env \
    --instance-type <gpu-type>

  /Users/area/heretic/scripts/phala_deploy.sh \
    --name heretic-onnx-debug \
    --env-file /Users/area/heretic/build/phala.env \
    --debug \
    --ssh-pubkey ~/.ssh/id_ed25519.pub
EOF
}

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
phala_cli="${repo_root}/scripts/phala_cli.sh"
compose_file="${repo_root}/docker/phala-compose.yml"
env_file=""
cvm_name=""
instance_type=""
region=""
ssh_pubkey=""
debug_mode="0"
link_project="1"
wait_for_completion="1"
public_logs="0"
public_sysinfo="0"
listed="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      cvm_name="$2"
      shift 2
      ;;
    --env-file)
      env_file="$2"
      shift 2
      ;;
    --instance-type)
      instance_type="$2"
      shift 2
      ;;
    --region)
      region="$2"
      shift 2
      ;;
    --compose)
      compose_file="$2"
      shift 2
      ;;
    --ssh-pubkey)
      ssh_pubkey="$2"
      shift 2
      ;;
    --debug)
      debug_mode="1"
      compose_file="${repo_root}/docker/phala-compose.debug.yml"
      shift
      ;;
    --link)
      link_project="1"
      shift
      ;;
    --no-link)
      link_project="0"
      shift
      ;;
    --wait)
      wait_for_completion="1"
      shift
      ;;
    --no-wait)
      wait_for_completion="0"
      shift
      ;;
    --public-logs)
      public_logs="1"
      shift
      ;;
    --public-sysinfo)
      public_sysinfo="1"
      shift
      ;;
    --listed)
      listed="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${cvm_name}" || -z "${env_file}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${env_file}" ]]; then
  echo "env file does not exist: ${env_file}" >&2
  exit 1
fi

if [[ ! -f "${compose_file}" ]]; then
  echo "compose file does not exist: ${compose_file}" >&2
  exit 1
fi

deploy_args=(
  deploy
  --name "${cvm_name}"
  --compose "${compose_file}"
  -e "${env_file}"
)

if [[ -n "${instance_type}" ]]; then
  deploy_args+=(--instance-type "${instance_type}")
fi

if [[ -n "${region}" ]]; then
  deploy_args+=(--region "${region}")
fi

if [[ "${wait_for_completion}" == "1" ]]; then
  deploy_args+=(--wait)
fi

if [[ "${public_logs}" == "1" ]]; then
  deploy_args+=(--public-logs)
else
  deploy_args+=(--no-public-logs)
fi

if [[ "${public_sysinfo}" == "1" ]]; then
  deploy_args+=(--public-sysinfo)
else
  deploy_args+=(--no-public-sysinfo)
fi

if [[ "${listed}" == "1" ]]; then
  deploy_args+=(--listed)
fi

if [[ "${debug_mode}" == "1" ]]; then
  if [[ -z "${ssh_pubkey}" ]]; then
    if [[ -f "${HOME}/.ssh/id_ed25519.pub" ]]; then
      ssh_pubkey="${HOME}/.ssh/id_ed25519.pub"
    elif [[ -f "${HOME}/.ssh/id_rsa.pub" ]]; then
      ssh_pubkey="${HOME}/.ssh/id_rsa.pub"
    else
      echo "debug mode requires an SSH public key; pass --ssh-pubkey <path>" >&2
      exit 1
    fi
  fi
  deploy_args+=(--dev-os --ssh-pubkey "${ssh_pubkey}")
fi

echo "running: ${phala_cli} ${deploy_args[*]}"
"${phala_cli}" "${deploy_args[@]}"

if [[ "${link_project}" == "1" ]]; then
  echo "linking repo to CVM ${cvm_name}"
  (
    cd "${repo_root}"
    "${phala_cli}" link "${cvm_name}"
  )
fi
