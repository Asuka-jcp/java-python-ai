#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_dataset(raw_data_root: Path, ct_name: str, suv_name: str, seg_name: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    records: list[dict[str, str]] = []
    missing_records: list[dict[str, str]] = []

    all_dirs = [p for p in raw_data_root.rglob("*") if p.is_dir()]
    logging.info("Scanning %d directories under %s", len(all_dirs), raw_data_root)

    for d in tqdm(all_dirs, desc="Inspecting directories"):
        ct_path = d / ct_name
        suv_path = d / suv_name
        seg_path = d / seg_name

        exists = {"ct": ct_path.exists(), "suv": suv_path.exists(), "seg": seg_path.exists()}
        if all(exists.values()):
            case_name = d.name
            patient_dir = d.parent.name if d.parent != d else ""
            study_dir = d.name
            records.append(
                {
                    "case_name": case_name,
                    "patient_dir": str(d.parent),
                    "study_dir": str(d),
                    "ctres_path": str(ct_path.resolve()),
                    "suv_path": str(suv_path.resolve()),
                    "seg_path": str(seg_path.resolve()),
                }
            )
        elif any(exists.values()):
            missing = [k for k, v in exists.items() if not v]
            missing_records.append({"dir": str(d), "missing": ",".join(missing)})

    manifest_df = pd.DataFrame(records)
    if not manifest_df.empty:
        manifest_df.insert(0, "case_index", range(1, len(manifest_df) + 1))
    return manifest_df, missing_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AutoPET dataset and build manifest.csv")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--output", type=Path, default=Path("manifest.csv"), help="Path to save manifest CSV")
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=Path("missing_files.csv"),
        help="Path to save directories with missing files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    try:
        cfg = load_config(args.config)
        raw_data_root = Path(cfg["raw_data_root"]).expanduser().resolve()
        if not raw_data_root.exists():
            raise FileNotFoundError(f"raw_data_root does not exist: {raw_data_root}")

        manifest_df, missing = inspect_dataset(
            raw_data_root=raw_data_root,
            ct_name=cfg["ct_filename"],
            suv_name=cfg["suv_filename"],
            seg_name=cfg["seg_filename"],
        )

        manifest_df.to_csv(args.output, index=False)
        logging.info("Found %d valid cases", len(manifest_df))
        logging.info("Saved manifest to: %s", args.output.resolve())

        if missing:
            pd.DataFrame(missing).to_csv(args.missing_output, index=False)
            logging.warning("Found %d directories with missing files", len(missing))
            logging.info("Saved missing report to: %s", args.missing_output.resolve())
        else:
            logging.info("No partially complete directories found")

    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to inspect dataset: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
