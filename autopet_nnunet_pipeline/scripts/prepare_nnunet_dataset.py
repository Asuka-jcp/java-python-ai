#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_prep import build_dataset_json, make_case_id  # noqa: E402


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def copy_case_files(df: pd.DataFrame, images_tr: Path, labels_tr: Path) -> int:
    count = 0
    for idx, row in df.iterrows():
        case_id = make_case_id(idx + 1)
        ct_dst = images_tr / f"{case_id}_0000.nii.gz"
        suv_dst = images_tr / f"{case_id}_0001.nii.gz"
        seg_dst = labels_tr / f"{case_id}.nii.gz"

        shutil.copy2(Path(row["ctres_path"]), ct_dst)
        shutil.copy2(Path(row["suv_path"]), suv_dst)
        shutil.copy2(Path(row["seg_path"]), seg_dst)
        count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare nnU-Net v2 raw dataset from manifest.csv")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"), help="Path to manifest.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    try:
        cfg = load_config(args.config)
        df = pd.read_csv(args.manifest)
        if df.empty:
            raise ValueError("manifest.csv is empty, no cases to prepare")

        nnunet_raw = Path(cfg["nnunet_raw"]).expanduser().resolve()
        dataset_name = cfg["dataset_name"]
        dataset_dir = nnunet_raw / dataset_name
        images_tr = dataset_dir / "imagesTr"
        labels_tr = dataset_dir / "labelsTr"
        images_tr.mkdir(parents=True, exist_ok=True)
        labels_tr.mkdir(parents=True, exist_ok=True)

        copied_count = copy_case_files(df, images_tr, labels_tr)

        dataset_json = build_dataset_json(
            channel_0_name=cfg["channel_0_name"],
            channel_1_name=cfg["channel_1_name"],
            label_name=cfg["label_name"],
            num_training=copied_count,
        )

        dataset_json_path = dataset_dir / "dataset.json"
        dataset_json_path.write_text(json.dumps(dataset_json, indent=2, ensure_ascii=False), encoding="utf-8")

        logging.info("Prepared dataset at %s", dataset_dir)
        logging.info("Copied %d cases", copied_count)
        logging.info("Saved dataset.json to %s", dataset_json_path)

    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to prepare nnU-Net dataset: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
