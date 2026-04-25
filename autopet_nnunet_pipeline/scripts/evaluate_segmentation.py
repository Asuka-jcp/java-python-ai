#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import dice_score, false_negative_volume, false_positive_volume  # noqa: E402


def load_array_and_voxel_volume(path: Path) -> tuple[np.ndarray, float]:
    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata())
    spacing = img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    return arr, voxel_volume_mm3


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions against labels")
    parser.add_argument("--prediction_dir", type=Path, required=True, help="Directory with predicted .nii.gz files")
    parser.add_argument("--label_dir", type=Path, required=True, help="Directory with ground truth .nii.gz files")
    parser.add_argument("--output_dir", type=Path, default=Path("."), help="Output directory for metrics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    try:
        pred_files = sorted(args.prediction_dir.glob("*.nii.gz"))
        if not pred_files:
            raise FileNotFoundError(f"No .nii.gz predictions found in: {args.prediction_dir}")

        rows = []
        for pred_path in pred_files:
            label_path = args.label_dir / pred_path.name
            if not label_path.exists():
                logging.warning("Missing label for prediction %s, skipping", pred_path.name)
                continue

            pred_arr, pred_voxel_vol = load_array_and_voxel_volume(pred_path)
            label_arr, _ = load_array_and_voxel_volume(label_path)

            if pred_arr.shape != label_arr.shape:
                logging.warning("Shape mismatch for %s, skipping", pred_path.name)
                continue

            dsc = dice_score(pred_arr, label_arr)
            fp_mm3 = false_positive_volume(pred_arr, label_arr, voxel_volume=pred_voxel_vol)
            fn_mm3 = false_negative_volume(pred_arr, label_arr, voxel_volume=pred_voxel_vol)

            rows.append(
                {
                    "case_name": pred_path.stem.replace(".nii", ""),
                    "dice": dsc,
                    "fp_volume_mm3": fp_mm3,
                    "fn_volume_mm3": fn_mm3,
                    "fp_volume_ml": fp_mm3 / 1000.0,
                    "fn_volume_ml": fn_mm3 / 1000.0,
                }
            )

        if not rows:
            raise RuntimeError("No valid prediction-label pairs were evaluated")

        out_dir = args.output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        per_case_csv = out_dir / "per_case_metrics.csv"
        pd.DataFrame(rows).to_csv(per_case_csv, index=False)

        summary = {
            "num_cases": len(rows),
            "dice_mean": float(np.mean([r["dice"] for r in rows])),
            "dice_std": float(np.std([r["dice"] for r in rows])),
            "fp_volume_ml_mean": float(np.mean([r["fp_volume_ml"] for r in rows])),
            "fn_volume_ml_mean": float(np.mean([r["fn_volume_ml"] for r in rows])),
        }

        summary_json = out_dir / "summary_metrics.json"
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        logging.info("Saved per-case metrics to %s", per_case_csv)
        logging.info("Saved summary metrics to %s", summary_json)

    except Exception as exc:  # noqa: BLE001
        logging.exception("Evaluation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
