#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.label_utils import flatten_case_labels, summarize_distributions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten region-level labels from JSON files")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    frames = []
    for _, row in manifest.iterrows():
        if not bool(row.get("has_json", False)):
            continue
        case_id = str(row["case_id"])
        json_path = Path(row["json_path"])
        if not json_path.exists():
            logging.warning("JSON file missing: %s", json_path)
            continue
        frames.append(flatten_case_labels(case_id, json_path))

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["case_id", "region", "pet_status", "suv", "ct_status", "ct_description"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.output, index=False)

    region_dist, pet_dist, ct_dist = summarize_distributions(all_df) if len(all_df) else (pd.Series(), pd.Series(), pd.Series())
    logging.info("Saved flattened labels: %s (%d rows)", args.output, len(all_df))
    logging.info("Total regions: %d", len(region_dist))
    logging.info("Region names: %s", list(region_dist.index))
    logging.info("pet_status distribution: %s", pet_dist.to_dict())
    logging.info("ct_status distribution: %s", ct_dist.to_dict())


if __name__ == "__main__":
    main()
