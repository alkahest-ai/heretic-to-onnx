#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-all}"

WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/build/phala_gpu_tee}"
MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/models}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$WORK_ROOT/manifests}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$WORK_ROOT/packages}"

DATASET_VARIANTS="${DATASET_VARIANTS:-25}"
DATASET_SEED="${DATASET_SEED:-111}"
PROMPT_LIMIT="${PROMPT_LIMIT:-240}"
DATASET_OUTPUT="${DATASET_OUTPUT:-$ROOT_DIR/data/roleplay_v1/generated/batch-0002.jsonl}"
DATASET_ID_PREFIX="${DATASET_ID_PREFIX:-bulk25}"

HF_UPLOAD_WORKERS="${HF_UPLOAD_WORKERS:-8}"
HF_PRIVATE="${HF_PRIVATE:-1}"

RALLY2_SOURCE_MODEL="${RALLY2_SOURCE_MODEL:-p-e-w/gemma-4-E2B-it-heretic-ara}"
RALLY4_SOURCE_MODEL="${RALLY4_SOURCE_MODEL:-coder3101/gemma-4-E4B-it-heretic}"
SHEENA_SOURCE_MODEL="${SHEENA_SOURCE_MODEL:-tvall43/Qwen3.5-4B-heretic}"

RALLY2_DIRECT_REPO="${RALLY2_DIRECT_REPO:-alkahest/rally-2b}"
RALLY4_DIRECT_REPO="${RALLY4_DIRECT_REPO:-alkahest/rally-4b}"
SHEENA_DIRECT_REPO="${SHEENA_DIRECT_REPO:-alkahest/sheena-4b}"
RALLY2_TUNED_REPO="${RALLY2_TUNED_REPO:-alkahest/rally-2b-rp}"
RALLY4_TUNED_REPO="${RALLY4_TUNED_REPO:-alkahest/rally-4b-rp}"
SHEENA_TUNED_REPO="${SHEENA_TUNED_REPO:-alkahest/sheena-4b-rp}"

RALLY_MAX_STEPS="${RALLY_MAX_STEPS:-300}"
RALLY4_MAX_STEPS="${RALLY4_MAX_STEPS:-250}"
SHEENA_MAX_STEPS="${SHEENA_MAX_STEPS:-300}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "$WORK_ROOT" "$MODEL_ROOT" "$MANIFEST_ROOT" "$PACKAGE_ROOT"

private_args=()
if [[ "$HF_PRIVATE" == "1" ]]; then
  private_args+=(--private)
fi

require_hf_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[error] HF_TOKEN must be set before running $1" >&2
    exit 1
  fi
}

bootstrap_env() {
  bash "$ROOT_DIR/scripts/phala_gpu_tee_bootstrap.sh"
}

build_dataset() {
  echo "[dataset] rendering prompt pack"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/render_roleplay_prompt_pack.py" --limit "$PROMPT_LIMIT"

  echo "[dataset] generating synthetic conversations"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/synthesize_roleplay_batch.py" \
    --variants "$DATASET_VARIANTS" \
    --seed "$DATASET_SEED" \
    --id-prefix "$DATASET_ID_PREFIX" \
    --output "$DATASET_OUTPUT"

  echo "[dataset] merging corpus"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/build_roleplay_training_corpus.py"

  echo "[dataset] validating and splitting corpus"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_roleplay_dataset.py" \
    --input "$ROOT_DIR/data/roleplay_v1/corpus.jsonl"
}

convert_direct() {
  local label="$1"
  local template="$2"
  local target_repo="$3"
  local manifest_path="$MANIFEST_ROOT/$label.yaml"
  local work_dir="$WORK_ROOT/$label"
  local package_dir="$PACKAGE_ROOT/$label"

  require_hf_token "$label direct conversion"

  echo "[manifest] $label"
  "$PYTHON_BIN" -m tools.heretic_to_onnx render-manifest \
    --template "$template" \
    --output "$manifest_path" \
    --target-repo-id "$target_repo"

  echo "[convert] $label"
  "$PYTHON_BIN" -m tools.heretic_to_onnx convert \
    --config "$manifest_path" \
    --work-dir "$work_dir" \
    --output-dir "$package_dir" \
    --force \
    --strict-onnx \
    --export-mode execute \
    --quantize-mode execute \
    --python-exec "$PYTHON_BIN"

  echo "[publish] $label"
  "$PYTHON_BIN" -m tools.heretic_to_onnx publish-hf \
    --config "$manifest_path" \
    --package-dir "$package_dir" \
    --num-workers "$HF_UPLOAD_WORKERS" \
    "${private_args[@]}"
}

train_model() {
  local label="$1"
  local model_name="$2"
  local output_dir="$3"
  local merged_dir="$4"
  local max_steps="$5"

  require_hf_token "$label training"

  echo "[train] $label"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_rally_unsloth.py" \
    --model-name "$model_name" \
    --train-file "$ROOT_DIR/data/roleplay_v1/splits/train.jsonl" \
    --val-file "$ROOT_DIR/data/roleplay_v1/splits/val.jsonl" \
    --output-dir "$output_dir" \
    --merged-output-dir "$merged_dir" \
    --max-steps "$max_steps" \
    --save-merged
}

convert_tuned_model() {
  local label="$1"
  local template="$2"
  local merged_dir="$3"
  local target_repo="$4"
  local manifest_path="$MANIFEST_ROOT/$label.yaml"
  local work_dir="$WORK_ROOT/$label"
  local package_dir="$PACKAGE_ROOT/$label"

  require_hf_token "$label ONNX packaging"

  echo "[manifest] $label"
  "$PYTHON_BIN" -m tools.heretic_to_onnx render-manifest \
    --template "$template" \
    --output "$manifest_path" \
    --source-model-id "$merged_dir" \
    --target-repo-id "$target_repo"

  echo "[convert] $label"
  "$PYTHON_BIN" -m tools.heretic_to_onnx convert \
    --config "$manifest_path" \
    --work-dir "$work_dir" \
    --output-dir "$package_dir" \
    --force \
    --strict-onnx \
    --export-mode execute \
    --quantize-mode execute \
    --python-exec "$PYTHON_BIN"

  echo "[publish] $label"
  "$PYTHON_BIN" -m tools.heretic_to_onnx publish-hf \
    --config "$manifest_path" \
    --package-dir "$package_dir" \
    --num-workers "$HF_UPLOAD_WORKERS" \
    "${private_args[@]}"
}

run_rally() {
  local output_dir="$MODEL_ROOT/rally-2b-rp"
  local merged_dir="$MODEL_ROOT/rally-2b-rp-merged"
  train_model "rally-2b-rp" "$RALLY2_SOURCE_MODEL" "$output_dir" "$merged_dir" "$RALLY_MAX_STEPS"
  convert_tuned_model \
    "rally-2b-rp" \
    "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" \
    "$merged_dir" \
    "$RALLY2_TUNED_REPO"
}

run_rally4() {
  local output_dir="$MODEL_ROOT/rally-4b-rp"
  local merged_dir="$MODEL_ROOT/rally-4b-rp-merged"
  train_model "rally-4b-rp" "$RALLY4_SOURCE_MODEL" "$output_dir" "$merged_dir" "$RALLY4_MAX_STEPS"
  convert_tuned_model \
    "rally-4b-rp" \
    "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml" \
    "$merged_dir" \
    "$RALLY4_TUNED_REPO"
}

run_sheena() {
  local output_dir="$MODEL_ROOT/sheena-4b-rp"
  local merged_dir="$MODEL_ROOT/sheena-4b-rp-merged"
  train_model "sheena-4b-rp" "$SHEENA_SOURCE_MODEL" "$output_dir" "$merged_dir" "$SHEENA_MAX_STEPS"
  convert_tuned_model \
    "sheena-4b-rp" \
    "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-4b-heretic.yaml" \
    "$merged_dir" \
    "$SHEENA_TUNED_REPO"
}

usage() {
  cat <<EOF
usage: bash scripts/phala_gpu_tee_oneclick.sh <mode>

modes:
  bootstrap
  dataset
  rally-2b-direct
  rally-4b-direct
  sheena-4b-direct
  rally
  rally-4b
  sheena
  all-gemma
  all
EOF
}

case "$MODE" in
  bootstrap)
    bootstrap_env
    ;;
  dataset)
    build_dataset
    ;;
  rally-2b-direct)
    convert_direct "rally-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" "$RALLY2_DIRECT_REPO"
    ;;
  rally-4b-direct)
    convert_direct "rally-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml" "$RALLY4_DIRECT_REPO"
    ;;
  sheena-4b-direct)
    convert_direct "sheena-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-4b-heretic.yaml" "$SHEENA_DIRECT_REPO"
    ;;
  rally)
    build_dataset
    run_rally
    ;;
  rally-4b)
    build_dataset
    run_rally4
    ;;
  sheena)
    build_dataset
    run_sheena
    ;;
  all-gemma)
    bootstrap_env
    build_dataset
    convert_direct "rally-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" "$RALLY2_DIRECT_REPO"
    convert_direct "rally-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml" "$RALLY4_DIRECT_REPO"
    run_rally
    run_rally4
    ;;
  all)
    bootstrap_env
    build_dataset
    convert_direct "rally-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" "$RALLY2_DIRECT_REPO"
    convert_direct "rally-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml" "$RALLY4_DIRECT_REPO"
    convert_direct "sheena-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-4b-heretic.yaml" "$SHEENA_DIRECT_REPO"
    run_rally
    run_rally4
    run_sheena
    ;;
  *)
    usage
    exit 1
    ;;
esac
