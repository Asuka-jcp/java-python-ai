#!/usr/bin/env bash
set -euo pipefail

FOLD="${1:-0}"
CONFIG="${2:-3d_fullres}"
DATASET_ID="${3:-501}"

echo "Running nnUNetv2_train with dataset=${DATASET_ID}, config=${CONFIG}, fold=${FOLD}"
nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD"
