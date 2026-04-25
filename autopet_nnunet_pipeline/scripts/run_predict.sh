#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
FOLD="${3:-0}"
CONFIG="${4:-3d_fullres}"
DATASET_ID="${5:-501}"

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 <input_dir> <output_dir> [fold=0] [config=3d_fullres] [dataset_id=501]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Running nnUNetv2_predict with dataset=${DATASET_ID}, config=${CONFIG}, fold=${FOLD}"
nnUNetv2_predict -i "$INPUT_DIR" -o "$OUTPUT_DIR" -d "$DATASET_ID" -c "$CONFIG" -f "$FOLD"
