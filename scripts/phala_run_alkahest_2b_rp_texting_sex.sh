#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SFT_DATASET_KIND="${SFT_DATASET_KIND:-hf_texting_sex}"
export SFT_DATASET_ID="${SFT_DATASET_ID:-Maxx0/Texting_sex}"
export SFT_DATASET_SPLIT="${SFT_DATASET_SPLIT:-train}"
export SFT_DATASET_OUTPUT_DIR="${SFT_DATASET_OUTPUT_DIR:-$ROOT_DIR/data/external_text_sft/texting_sex}"
export SFT_DATASET_VAL_FRACTION="${SFT_DATASET_VAL_FRACTION:-0.02}"
export SFT_DATASET_MAX_ROWS="${SFT_DATASET_MAX_ROWS:-0}"
export SFT_DATASET_MIN_MESSAGE_CHARS="${SFT_DATASET_MIN_MESSAGE_CHARS:-80}"

bash "$ROOT_DIR/scripts/phala_gpu_tee_oneclick.sh" alkahest-2b
