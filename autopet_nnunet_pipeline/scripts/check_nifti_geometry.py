#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nifti_utils import check_binary_label, get_affine, get_shape, get_spacing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check geometry consistency for CTres/SUV/SEG")
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"), help="Path to manifest.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("geometry_report.csv"),
        help="Path to save geometry report",
    )
    parser.add_argument("--affine-atol", type=float, default=1e-4, help="Absolute tolerance for affine comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    try:
        df = pd.read_csv(args.manifest)
        anomalies: list[dict[str, str]] = []

        for _, row in df.iterrows():
            case_name = row["case_name"]
            ct_path = Path(row["ctres_path"])
            suv_path = Path(row["suv_path"])
            seg_path = Path(row["seg_path"])

            issues = []
            ct_shape, suv_shape, seg_shape = get_shape(ct_path), get_shape(suv_path), get_shape(seg_path)
            if not (ct_shape == suv_shape == seg_shape):
                issues.append(f"shape_mismatch: ct={ct_shape}, suv={suv_shape}, seg={seg_shape}")

            ct_spacing, suv_spacing, seg_spacing = get_spacing(ct_path), get_spacing(suv_path), get_spacing(seg_path)
            if not (ct_spacing == suv_spacing == seg_spacing):
                issues.append(f"spacing_mismatch: ct={ct_spacing}, suv={suv_spacing}, seg={seg_spacing}")

            ct_affine, suv_affine, seg_affine = get_affine(ct_path), get_affine(suv_path), get_affine(seg_path)
            if not (np.allclose(ct_affine, suv_affine, atol=args.affine_atol) and np.allclose(ct_affine, seg_affine, atol=args.affine_atol)):
                issues.append("affine_mismatch")

            if not check_binary_label(seg_path):
                issues.append("seg_not_binary")

            anomalies.append(
                {
                    "case_name": case_name,
                    "ctres_path": str(ct_path),
                    "suv_path": str(suv_path),
                    "seg_path": str(seg_path),
                    "is_ok": len(issues) == 0,
                    "issues": " | ".join(issues),
                }
            )

        report_df = pd.DataFrame(anomalies)
        report_df.to_csv(args.output, index=False)
        abnormal_df = report_df[report_df["is_ok"] == False]  # noqa: E712

        logging.info("Checked %d cases", len(report_df))
        logging.info("Anomalies: %d", len(abnormal_df))
        logging.info("Saved report to %s", args.output.resolve())

    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to check geometry: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
