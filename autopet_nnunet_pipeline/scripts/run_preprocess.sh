#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/config.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

# Parse YAML with Python for portability.
# shellcheck disable=SC1090
source <(python - <<'PY' "$CONFIG_PATH"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))

print(f"export nnUNet_raw='{Path(cfg['nnunet_raw']).expanduser().resolve()}'")
print(f"export nnUNet_preprocessed='{Path(cfg['nnunet_preprocessed']).expanduser().resolve()}'")
print(f"export nnUNet_results='{Path(cfg['nnunet_results']).expanduser().resolve()}'")
print(f"export DATASET_ID='{cfg['dataset_id']}'")
PY
)

echo "Using nnUNet_raw=${nnUNet_raw}"
echo "Using nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "Using nnUNet_results=${nnUNet_results}"

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity
