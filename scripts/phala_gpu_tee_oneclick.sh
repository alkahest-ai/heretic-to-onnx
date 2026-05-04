#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-python3}"
MODE="${1:-all}"

WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/build/phala_gpu_tee}"
MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/models}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$WORK_ROOT/manifests}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$WORK_ROOT/packages}"

DATASET_COUNT="${DATASET_COUNT:-300}"
DATASET_SEED="${DATASET_SEED:-111}"
PROMPT_LIMIT="${PROMPT_LIMIT:-300}"
DATASET_OUTPUT="${DATASET_OUTPUT:-$ROOT_DIR/data/roleplay_v2/generated_raw/batch-0001.jsonl}"
REVIEW_OUTPUT="${REVIEW_OUTPUT:-$ROOT_DIR/data/roleplay_v2/review_table/batch-0001.tsv}"
DATASET_ID_PREFIX="${DATASET_ID_PREFIX:-v2b001}"
DATASET_BATCH_ID="${DATASET_BATCH_ID:-batch-0001}"
MIN_APPROVED_ROWS="${MIN_APPROVED_ROWS:-5000}"
SFT_DATASET_KIND="${SFT_DATASET_KIND:-roleplay_v2}"
SFT_DATASET_ID="${SFT_DATASET_ID:-Maxx0/Texting_sex}"
SFT_DATASET_SPLIT="${SFT_DATASET_SPLIT:-train}"
SFT_DATASET_OUTPUT_DIR="${SFT_DATASET_OUTPUT_DIR:-$ROOT_DIR/data/external_text_sft/texting_sex}"
SFT_DATASET_VAL_FRACTION="${SFT_DATASET_VAL_FRACTION:-0.02}"
SFT_DATASET_MAX_ROWS="${SFT_DATASET_MAX_ROWS:-0}"
SFT_DATASET_MIN_MESSAGE_CHARS="${SFT_DATASET_MIN_MESSAGE_CHARS:-80}"

HF_UPLOAD_WORKERS="${HF_UPLOAD_WORKERS:-8}"
HF_PRIVATE="${HF_PRIVATE:-1}"
HF_OWNER="${HF_OWNER:-alkahest-ai}"

SOURCE_MODE="${SOURCE_MODE:-prebuilt_heretic}"
HERETIC_OUTPUT_ROOT="${HERETIC_OUTPUT_ROOT:-$MODEL_ROOT/heretic}"
RALLY2_BASE_MODEL="${RALLY2_BASE_MODEL:-google/gemma-4-E2B-it}"
ALKAHEST2_BASE_MODEL="${ALKAHEST2_BASE_MODEL:-Qwen/Qwen3.5-2B}"
RALLY2_SOURCE_MODEL="${RALLY2_SOURCE_MODEL:-p-e-w/gemma-4-E2B-it-heretic-ara}"
RALLY4_SOURCE_MODEL="${RALLY4_SOURCE_MODEL:-coder3101/gemma-4-E4B-it-heretic}"
ALKAHEST4_SOURCE_MODEL="${ALKAHEST4_SOURCE_MODEL:-tvall43/Qwen3.5-4B-heretic}"
ALKAHEST2_SOURCE_MODEL="${ALKAHEST2_SOURCE_MODEL:-tvall43/Qwen3.5-2B-heretic-v3b}"
ALKAHEST08_SOURCE_MODEL="${ALKAHEST08_SOURCE_MODEL:-tvall43/Qwen3.5-0.8B-heretic-v3}"

RALLY2_DIRECT_REPO="${RALLY2_DIRECT_REPO:-${HF_OWNER}/rally-2b}"
RALLY2_DIRECT_TEXT_REPO="${RALLY2_DIRECT_TEXT_REPO:-${HF_OWNER}/rally-2b-text}"
RALLY4_DIRECT_REPO="${RALLY4_DIRECT_REPO:-${HF_OWNER}/rally-4b}"
ALKAHEST4_DIRECT_REPO="${ALKAHEST4_DIRECT_REPO:-${HF_OWNER}/alkahest-4b}"
ALKAHEST2_DIRECT_REPO="${ALKAHEST2_DIRECT_REPO:-${HF_OWNER}/alkahest-2b}"
ALKAHEST08_DIRECT_REPO="${ALKAHEST08_DIRECT_REPO:-${HF_OWNER}/alkahest-0.8b}"
RALLY2_V2_DIRECT_REPO="${RALLY2_V2_DIRECT_REPO:-${HF_OWNER}/rally-2b-v2}"
RALLY4_V2_DIRECT_REPO="${RALLY4_V2_DIRECT_REPO:-${HF_OWNER}/rally-4b-v2}"
ALKAHEST4_V2_DIRECT_REPO="${ALKAHEST4_V2_DIRECT_REPO:-${HF_OWNER}/alkahest-4b-v2}"
ALKAHEST2_V2_DIRECT_REPO="${ALKAHEST2_V2_DIRECT_REPO:-${HF_OWNER}/alkahest-2b-v2}"
ALKAHEST08_V2_DIRECT_REPO="${ALKAHEST08_V2_DIRECT_REPO:-${HF_OWNER}/alkahest-0.8b-v2}"
RALLY2_TUNED_REPO="${RALLY2_TUNED_REPO:-${HF_OWNER}/rally-2b-rp}"
RALLY2_TUNED_TEXT_REPO="${RALLY2_TUNED_TEXT_REPO:-${HF_OWNER}/rally-2b-rp-text}"
RALLY2_MERGED_REPO="${RALLY2_MERGED_REPO:-${HF_OWNER}/rally-2b-rp-a100-b75-merged}"
RALLY4_TUNED_REPO="${RALLY4_TUNED_REPO:-${HF_OWNER}/rally-4b-rp}"
ALKAHEST4_TUNED_REPO="${ALKAHEST4_TUNED_REPO:-${HF_OWNER}/alkahest-4b-rp}"
ALKAHEST2_TUNED_REPO="${ALKAHEST2_TUNED_REPO:-${HF_OWNER}/alkahest-2b-rp}"
ALKAHEST08_TUNED_REPO="${ALKAHEST08_TUNED_REPO:-${HF_OWNER}/alkahest-0.8b-rp}"

RALLY_MAX_STEPS="${RALLY_MAX_STEPS:-300}"
RALLY_STAGE_A_MAX_STEPS="${RALLY_STAGE_A_MAX_STEPS:-300}"
RALLY_STAGE_B_MAX_STEPS="${RALLY_STAGE_B_MAX_STEPS:-450}"
RALLY_TWO_STAGE_DATA_DIR="${RALLY_TWO_STAGE_DATA_DIR:-$WORK_ROOT/rally_two_stage_sft_v8}"
RALLY_STAGE_A_REPEATS="${RALLY_STAGE_A_REPEATS:-18}"
RALLY_STAGE_B_BOUNDARY_REPEATS="${RALLY_STAGE_B_BOUNDARY_REPEATS:-80}"
RALLY_STAGE_B_ADULT_REPEATS="${RALLY_STAGE_B_ADULT_REPEATS:-40}"
RALLY2_RP_STAGE_B_SCALE="${RALLY2_RP_STAGE_B_SCALE:-0.75}"
RALLY4_MAX_STEPS="${RALLY4_MAX_STEPS:-250}"
ALKAHEST4_MAX_STEPS="${ALKAHEST4_MAX_STEPS:-300}"
ALKAHEST2_MAX_STEPS="${ALKAHEST2_MAX_STEPS:-325}"
ALKAHEST08_MAX_STEPS="${ALKAHEST08_MAX_STEPS:-350}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

case "$SOURCE_MODE" in
  prebuilt_heretic|base_plus_heretic)
    ;;
  *)
    echo "[error] unsupported SOURCE_MODE: $SOURCE_MODE" >&2
    echo "[error] expected prebuilt_heretic or base_plus_heretic" >&2
    exit 1
    ;;
esac

mkdir -p "$WORK_ROOT" "$MODEL_ROOT" "$MANIFEST_ROOT" "$PACKAGE_ROOT"

TRAIN_FILE="${ROOT_DIR}/data/roleplay_v2/splits/train.jsonl"
VAL_FILE="${ROOT_DIR}/data/roleplay_v2/splits/val.jsonl"
DATASET_MANIFEST_PATH="${ROOT_DIR}/data/roleplay_v2/splits/manifest.json"

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

require_merged_checkpoint() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path/config.json" ]]; then
    echo "[error] SOURCE_MODE=base_plus_heretic needs $label merged checkpoint at $path" >&2
    echo "[error] run scripts/kaggle_heretic_2b_proof.py --label ${label%%-*}-2b first, then set HERETIC_OUTPUT_ROOT" >&2
    exit 1
  fi
  if [[ ! -f "$path/generation_config.json" ]]; then
    echo "[error] $label merged checkpoint is missing generation_config.json: $path" >&2
    exit 1
  fi
  if [[ ! -f "$path/tokenizer_config.json" ]]; then
    echo "[error] $label merged checkpoint is missing tokenizer_config.json: $path" >&2
    exit 1
  fi
  if [[ ! -f "$path/tokenizer.json" && ! -f "$path/tokenizer.model" && ! -f "$path/vocab.json" ]]; then
    echo "[error] $label merged checkpoint is missing tokenizer assets: $path" >&2
    exit 1
  fi
  if ! find "$path" -maxdepth 1 \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) | grep -q .; then
    echo "[error] $label merged checkpoint has no model weights: $path" >&2
    exit 1
  fi
}

resolve_heretic_source() {
  local label="$1"
  case "$label" in
    rally-2b-direct|rally-2b-v2-direct|rally-2b-rp)
      HERETIC_SOURCE_MODEL="$HERETIC_OUTPUT_ROOT/rally-2b-heretic-merged"
      HERETIC_BASE_MODEL="$RALLY2_BASE_MODEL"
      require_merged_checkpoint "rally-2b" "$HERETIC_SOURCE_MODEL"
      ;;
    alkahest-2b-direct|alkahest-2b-v2-direct|alkahest-2b-rp)
      HERETIC_SOURCE_MODEL="$HERETIC_OUTPUT_ROOT/alkahest-2b-heretic-merged"
      HERETIC_BASE_MODEL="$ALKAHEST2_BASE_MODEL"
      require_merged_checkpoint "alkahest-2b" "$HERETIC_SOURCE_MODEL"
      ;;
    *)
      HERETIC_SOURCE_MODEL=""
      HERETIC_BASE_MODEL=""
      ;;
  esac
}

bootstrap_env() {
  bash "$ROOT_DIR/scripts/phala_gpu_tee_bootstrap.sh"
}

generate_review_batch() {
  echo "[dataset] rendering roleplay_v2 prompt pack"
  "$TRAIN_PYTHON_BIN" "$ROOT_DIR/scripts/render_roleplay_prompt_pack.py" --limit "$PROMPT_LIMIT"

  echo "[dataset] generating reviewed-batch candidate conversations"
  "$TRAIN_PYTHON_BIN" "$ROOT_DIR/scripts/synthesize_roleplay_batch.py" \
    --count "$DATASET_COUNT" \
    --seed "$DATASET_SEED" \
    --id-prefix "$DATASET_ID_PREFIX" \
    --batch-id "$DATASET_BATCH_ID" \
    --output "$DATASET_OUTPUT" \
    --review-output "$REVIEW_OUTPUT"
}

compile_approved_dataset() {
  local approved_dir="$ROOT_DIR/data/roleplay_v2/approved_jsonl"
  local approved_files
  local approved_rows

  approved_files="$(find "$approved_dir" -maxdepth 1 -name '*.jsonl' | wc -l | tr -d ' ')"
  if [[ "$approved_files" == "0" ]]; then
    echo "[dataset] no approved roleplay_v2 JSONL files found in $approved_dir" >&2
    echo "[dataset] generate and review a batch first, then compile it with review_table_to_jsonl.py" >&2
    exit 1
  fi

  approved_rows="$(find "$approved_dir" -maxdepth 1 -name '*.jsonl' -exec cat {} + | wc -l | tr -d ' ')"
  if [[ "$approved_rows" -lt "$MIN_APPROVED_ROWS" ]]; then
    echo "[dataset] approved corpus has $approved_rows conversations; MIN_APPROVED_ROWS is $MIN_APPROVED_ROWS" >&2
    echo "[dataset] lower MIN_APPROVED_ROWS only if you intentionally want a smoke-test tune" >&2
    exit 1
  fi

  echo "[dataset] merging approved roleplay_v2 corpus"
  "$TRAIN_PYTHON_BIN" "$ROOT_DIR/scripts/build_roleplay_training_corpus.py"

  echo "[dataset] validating and splitting approved corpus"
  "$TRAIN_PYTHON_BIN" "$ROOT_DIR/scripts/prepare_roleplay_dataset.py" \
    --input "$ROOT_DIR/data/roleplay_v2/corpus.jsonl"
}

prepare_training_dataset() {
  case "$SFT_DATASET_KIND" in
    roleplay_v2)
      compile_approved_dataset
      TRAIN_FILE="$ROOT_DIR/data/roleplay_v2/splits/train.jsonl"
      VAL_FILE="$ROOT_DIR/data/roleplay_v2/splits/val.jsonl"
      DATASET_MANIFEST_PATH="$ROOT_DIR/data/roleplay_v2/splits/manifest.json"
      ;;
    hf_texting_sex)
      echo "[dataset] preparing external chat SFT dataset from $SFT_DATASET_ID"
      "$TRAIN_PYTHON_BIN" "$ROOT_DIR/scripts/prepare_texting_sex_dataset.py" \
        --dataset-id "$SFT_DATASET_ID" \
        --split "$SFT_DATASET_SPLIT" \
        --output-dir "$SFT_DATASET_OUTPUT_DIR" \
        --val-fraction "$SFT_DATASET_VAL_FRACTION" \
        --max-rows "$SFT_DATASET_MAX_ROWS" \
        --min-message-chars "$SFT_DATASET_MIN_MESSAGE_CHARS"
      TRAIN_FILE="$SFT_DATASET_OUTPUT_DIR/train.jsonl"
      VAL_FILE="$SFT_DATASET_OUTPUT_DIR/val.jsonl"
      DATASET_MANIFEST_PATH="$SFT_DATASET_OUTPUT_DIR/manifest.json"
      ;;
    *)
      echo "[error] unsupported SFT_DATASET_KIND: $SFT_DATASET_KIND" >&2
      exit 1
      ;;
  esac
}

convert_direct() {
  local label="$1"
  local template="$2"
  local target_repo="$3"
  local manifest_path="$MANIFEST_ROOT/$label.yaml"
  local work_dir="$WORK_ROOT/$label"
  local package_dir="$PACKAGE_ROOT/$label"
  local source_override=""
  local base_override=""

  require_hf_token "$label direct conversion"

  if [[ "$SOURCE_MODE" == "base_plus_heretic" ]]; then
    resolve_heretic_source "$label"
    if [[ -n "$HERETIC_SOURCE_MODEL" ]]; then
      source_override="$HERETIC_SOURCE_MODEL"
      base_override="$HERETIC_BASE_MODEL"
      echo "[manifest] using local Heretic checkpoint $source_override"
    else
      echo "[error] SOURCE_MODE=base_plus_heretic currently supports only rally-2b and alkahest-2b proof labels" >&2
      echo "[error] unsupported direct label: $label" >&2
      exit 1
    fi
  elif [[ "$label" == *"-v2-direct" ]]; then
    local sibling_label="${label%-v2-direct}-direct"
    local sibling_source="$WORK_ROOT/$sibling_label/inputs/source"
    if [[ -f "$sibling_source/config.json" ]] && [[ -f "$sibling_source/model.safetensors.index.json" || -f "$sibling_source/model.safetensors" ]]; then
      source_override="$sibling_source"
      echo "[manifest] reusing existing source snapshot from $source_override"
    fi
  fi

  echo "[manifest] $label"
  render_args=(
    "$PYTHON_BIN" -m tools.heretic_to_onnx render-manifest
    --template "$template"
    --output "$manifest_path"
    --target-repo-id "$target_repo"
  )
  if [[ -n "$source_override" ]]; then
    render_args+=(--source-model-id "$source_override")
  fi
  if [[ -n "$base_override" ]]; then
    render_args+=(--base-model-id "$base_override")
  fi
  "${render_args[@]}"

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
  "$TRAIN_PYTHON_BIN" "$ROOT_DIR/scripts/train_rally_unsloth.py" \
    --model-name "$model_name" \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --dataset-manifest "$DATASET_MANIFEST_PATH" \
    --output-dir "$output_dir" \
    --merged-output-dir "$merged_dir" \
    --max-steps "$max_steps" \
    --save-merged
}

prepare_rally_two_stage_dataset() {
  echo "[dataset] preparing Rally E2B two-stage RP v8 splits"
  "$TRAIN_PYTHON_BIN" "$ROOT_DIR/scripts/prepare_alkahest_two_stage_sft.py" \
    --output-dir "$RALLY_TWO_STAGE_DATA_DIR" \
    --stage-a-repeats "$RALLY_STAGE_A_REPEATS" \
    --stage-b-boundary-repeats "$RALLY_STAGE_B_BOUNDARY_REPEATS" \
    --stage-b-adult-repeats "$RALLY_STAGE_B_ADULT_REPEATS"
}

merge_scaled_candidate() {
  local base_dir="$1"
  local adapter_dir="$2"
  local output_dir="$3"
  local scale="$4"

  echo "[merge] scaled candidate $output_dir with scale=$scale"
  "$TRAIN_PYTHON_BIN" "$ROOT_DIR/scripts/merge_lora_scaled.py" \
    --base-dir "$base_dir" \
    --adapter-dir "$adapter_dir" \
    --output-dir "$output_dir" \
    --scale "$scale"
}

publish_merged_checkpoint() {
  local label="$1"
  local merged_dir="$2"
  local target_repo="$3"
  local create_args

  require_hf_token "$label merged checkpoint upload"
  create_args=(hf repos create "$target_repo" --type model --exist-ok)
  if [[ "$HF_PRIVATE" == "1" ]]; then
    create_args+=(--private)
  fi

  echo "[publish] $label merged checkpoint -> $target_repo"
  "${create_args[@]}"
  hf upload-large-folder \
    "$target_repo" \
    "$merged_dir" \
    --type model \
    --num-workers "$HF_UPLOAD_WORKERS" \
    --no-bars
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

run_rally_legacy() {
  local output_dir="$MODEL_ROOT/rally-2b-rp"
  local merged_dir="$MODEL_ROOT/rally-2b-rp-merged"
  local source_model="$RALLY2_SOURCE_MODEL"
  if [[ "$SOURCE_MODE" == "base_plus_heretic" ]]; then
    resolve_heretic_source "rally-2b-rp"
    source_model="$HERETIC_SOURCE_MODEL"
  fi
  train_model "rally-2b-rp" "$source_model" "$output_dir" "$merged_dir" "$RALLY_MAX_STEPS"
  convert_tuned_model \
    "rally-2b-rp" \
    "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara-v2.yaml" \
    "$merged_dir" \
    "$RALLY2_TUNED_REPO"
}

run_rally() {
  local stage_a_adapter="$MODEL_ROOT/rally-2b-rp-stage-a-adapter"
  local stage_a_merged="$MODEL_ROOT/rally-2b-rp-stage-a-merged"
  local stage_b_adapter="$MODEL_ROOT/rally-2b-rp-stage-b-adapter"
  local stage_ab_merged="$MODEL_ROOT/rally-2b-rp-stage-ab-merged"
  local selected_merged="$MODEL_ROOT/rally-2b-rp-a100-b75-merged"
  local source_model="$RALLY2_SOURCE_MODEL"

  if [[ "$SOURCE_MODE" == "base_plus_heretic" ]]; then
    resolve_heretic_source "rally-2b-rp"
    source_model="$HERETIC_SOURCE_MODEL"
  fi

  prepare_rally_two_stage_dataset

  TRAIN_FILE="$RALLY_TWO_STAGE_DATA_DIR/stage_a/train.jsonl"
  VAL_FILE="$RALLY_TWO_STAGE_DATA_DIR/stage_a/val.jsonl"
  DATASET_MANIFEST_PATH="$RALLY_TWO_STAGE_DATA_DIR/manifest.json"
  train_model "rally-2b-rp-stage-a" "$source_model" "$stage_a_adapter" "$stage_a_merged" "$RALLY_STAGE_A_MAX_STEPS"

  TRAIN_FILE="$RALLY_TWO_STAGE_DATA_DIR/stage_b/train.jsonl"
  VAL_FILE="$RALLY_TWO_STAGE_DATA_DIR/stage_b/val.jsonl"
  DATASET_MANIFEST_PATH="$RALLY_TWO_STAGE_DATA_DIR/manifest.json"
  train_model "rally-2b-rp-stage-b" "$stage_a_merged" "$stage_b_adapter" "$stage_ab_merged" "$RALLY_STAGE_B_MAX_STEPS"

  merge_scaled_candidate "$stage_a_merged" "$stage_b_adapter" "$selected_merged" "$RALLY2_RP_STAGE_B_SCALE"
  publish_merged_checkpoint "rally-2b-rp-a100-b75" "$selected_merged" "$RALLY2_MERGED_REPO"

  convert_tuned_model \
    "rally-2b-rp-a100-b75-full" \
    "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" \
    "$selected_merged" \
    "$RALLY2_TUNED_REPO"
  convert_tuned_model \
    "rally-2b-rp-a100-b75-text" \
    "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-rp-text.yaml" \
    "$selected_merged" \
    "$RALLY2_TUNED_TEXT_REPO"
}

run_rally4() {
  if [[ "$SOURCE_MODE" == "base_plus_heretic" ]]; then
    echo "[error] SOURCE_MODE=base_plus_heretic does not support rally-4b-rp yet" >&2
    exit 1
  fi
  local output_dir="$MODEL_ROOT/rally-4b-rp"
  local merged_dir="$MODEL_ROOT/rally-4b-rp-merged"
  train_model "rally-4b-rp" "$RALLY4_SOURCE_MODEL" "$output_dir" "$merged_dir" "$RALLY4_MAX_STEPS"
  convert_tuned_model \
    "rally-4b-rp" \
    "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml" \
    "$merged_dir" \
    "$RALLY4_TUNED_REPO"
}

run_alkahest4() {
  if [[ "$SOURCE_MODE" == "base_plus_heretic" ]]; then
    echo "[error] SOURCE_MODE=base_plus_heretic does not support alkahest-4b-rp yet" >&2
    exit 1
  fi
  local output_dir="$MODEL_ROOT/alkahest-4b-rp"
  local merged_dir="$MODEL_ROOT/alkahest-4b-rp-merged"
  train_model "alkahest-4b-rp" "$ALKAHEST4_SOURCE_MODEL" "$output_dir" "$merged_dir" "$ALKAHEST4_MAX_STEPS"
  convert_tuned_model \
    "alkahest-4b-rp" \
    "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-4b-heretic.yaml" \
    "$merged_dir" \
    "$ALKAHEST4_TUNED_REPO"
}

run_alkahest2() {
  local output_dir="$MODEL_ROOT/alkahest-2b-rp"
  local merged_dir="$MODEL_ROOT/alkahest-2b-rp-merged"
  local source_model="$ALKAHEST2_SOURCE_MODEL"
  if [[ "$SOURCE_MODE" == "base_plus_heretic" ]]; then
    resolve_heretic_source "alkahest-2b-rp"
    source_model="$HERETIC_SOURCE_MODEL"
  fi
  train_model "alkahest-2b-rp" "$source_model" "$output_dir" "$merged_dir" "$ALKAHEST2_MAX_STEPS"
  convert_tuned_model \
    "alkahest-2b-rp" \
    "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-2b-heretic-v2.yaml" \
    "$merged_dir" \
    "$ALKAHEST2_TUNED_REPO"
}

run_alkahest08() {
  if [[ "$SOURCE_MODE" == "base_plus_heretic" ]]; then
    echo "[error] SOURCE_MODE=base_plus_heretic does not support alkahest-0.8b-rp yet" >&2
    exit 1
  fi
  local output_dir="$MODEL_ROOT/alkahest-0.8b-rp"
  local merged_dir="$MODEL_ROOT/alkahest-0.8b-rp-merged"
  train_model "alkahest-0.8b-rp" "$ALKAHEST08_SOURCE_MODEL" "$output_dir" "$merged_dir" "$ALKAHEST08_MAX_STEPS"
  convert_tuned_model \
    "alkahest-0.8b-rp" \
    "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-0.8b-heretic.yaml" \
    "$merged_dir" \
    "$ALKAHEST08_TUNED_REPO"
}

usage() {
  cat <<EOF
usage: bash scripts/phala_gpu_tee_oneclick.sh <mode>

modes:
  bootstrap
  dataset
  dataset-batch
  dataset-compile
  rally-2b-direct
  rally-2b-text
  rally-4b-direct
  alkahest-4b-direct
  alkahest-2b-direct
  alkahest-0.8b-direct
  rally-2b-v2-direct
  rally-4b-v2-direct
  alkahest-4b-v2-direct
  alkahest-2b-v2-direct
  alkahest-0.8b-v2-direct
  rally
  rally-legacy
  rally-e2b
  rally-4b
  alkahest-4b
  alkahest-2b
  alkahest-0.8b
  all-gemma
  all-qwen
  all
EOF
}

case "$MODE" in
  bootstrap)
    bootstrap_env
    ;;
  dataset)
    generate_review_batch
    ;;
  dataset-batch)
    generate_review_batch
    ;;
  dataset-compile)
    compile_approved_dataset
    ;;
  rally-2b-direct)
    convert_direct "rally-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" "$RALLY2_DIRECT_REPO"
    ;;
  rally-2b-text)
    convert_direct "rally-2b-text" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara-text.yaml" "$RALLY2_DIRECT_TEXT_REPO"
    ;;
  rally-4b-direct)
    convert_direct "rally-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml" "$RALLY4_DIRECT_REPO"
    ;;
  alkahest-4b-direct)
    convert_direct "alkahest-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-4b-heretic.yaml" "$ALKAHEST4_DIRECT_REPO"
    ;;
  alkahest-2b-direct)
    convert_direct "alkahest-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-2b-heretic.yaml" "$ALKAHEST2_DIRECT_REPO"
    ;;
  alkahest-0.8b-direct)
    convert_direct "alkahest-0.8b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-0.8b-heretic.yaml" "$ALKAHEST08_DIRECT_REPO"
    ;;
  rally-2b-v2-direct)
    convert_direct "rally-2b-v2-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara-v2.yaml" "$RALLY2_V2_DIRECT_REPO"
    ;;
  rally-4b-v2-direct)
    convert_direct "rally-4b-v2-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic-v2.yaml" "$RALLY4_V2_DIRECT_REPO"
    ;;
  alkahest-4b-v2-direct)
    convert_direct "alkahest-4b-v2-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-4b-heretic-v2.yaml" "$ALKAHEST4_V2_DIRECT_REPO"
    ;;
  alkahest-2b-v2-direct)
    convert_direct "alkahest-2b-v2-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-2b-heretic-v2.yaml" "$ALKAHEST2_V2_DIRECT_REPO"
    ;;
  alkahest-0.8b-v2-direct)
    convert_direct "alkahest-0.8b-v2-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-0.8b-heretic-v2.yaml" "$ALKAHEST08_V2_DIRECT_REPO"
    ;;
  rally)
    run_rally
    ;;
  rally-legacy)
    prepare_training_dataset
    run_rally_legacy
    ;;
  rally-e2b)
    bootstrap_env
    convert_direct "rally-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" "$RALLY2_DIRECT_REPO"
    convert_direct "rally-2b-text" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara-text.yaml" "$RALLY2_DIRECT_TEXT_REPO"
    run_rally
    ;;
  rally-4b)
    compile_approved_dataset
    run_rally4
    ;;
  alkahest-4b)
    compile_approved_dataset
    run_alkahest4
    ;;
  alkahest-2b)
    prepare_training_dataset
    run_alkahest2
    ;;
  alkahest-0.8b)
    compile_approved_dataset
    run_alkahest08
    ;;
  all-gemma)
    bootstrap_env
    convert_direct "rally-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" "$RALLY2_DIRECT_REPO"
    convert_direct "rally-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml" "$RALLY4_DIRECT_REPO"
    compile_approved_dataset
    run_rally
    run_rally4
    ;;
  all-qwen)
    bootstrap_env
    convert_direct "alkahest-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-4b-heretic.yaml" "$ALKAHEST4_DIRECT_REPO"
    convert_direct "alkahest-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-2b-heretic.yaml" "$ALKAHEST2_DIRECT_REPO"
    convert_direct "alkahest-0.8b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-0.8b-heretic.yaml" "$ALKAHEST08_DIRECT_REPO"
    compile_approved_dataset
    run_alkahest4
    run_alkahest2
    run_alkahest08
    ;;
  all)
    bootstrap_env
    convert_direct "rally-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e2b-heretic-ara.yaml" "$RALLY2_DIRECT_REPO"
    convert_direct "rally-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.gemma4-e4b-heretic.yaml" "$RALLY4_DIRECT_REPO"
    convert_direct "alkahest-4b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-4b-heretic.yaml" "$ALKAHEST4_DIRECT_REPO"
    convert_direct "alkahest-2b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-2b-heretic.yaml" "$ALKAHEST2_DIRECT_REPO"
    convert_direct "alkahest-0.8b-direct" "$ROOT_DIR/configs/heretic-to-onnx.qwen3-5-0.8b-heretic.yaml" "$ALKAHEST08_DIRECT_REPO"
    compile_approved_dataset
    run_rally
    run_rally4
    run_alkahest4
    run_alkahest2
    run_alkahest08
    ;;
  *)
    usage
    exit 1
    ;;
esac
