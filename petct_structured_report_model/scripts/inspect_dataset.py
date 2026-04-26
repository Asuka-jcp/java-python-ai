#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan PET/CT dataset and build manifest.csv")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_manifest(data_root: Path) -> pd.DataFrame:
    cases = {}
    for nii_path in data_root.rglob("*.nii.gz"):
        stem = nii_path.name.replace(".nii.gz", "")
        if stem.endswith("_0000"):
            cid = stem[:-5]
            cases.setdefault(cid, {})["ct_path"] = nii_path
        elif stem.endswith("_0001"):
            cid = stem[:-5]
            cases.setdefault(cid, {})["pet_path"] = nii_path
    for json_path in data_root.rglob("*.json"):
        cid = json_path.stem
        cases.setdefault(cid, {})["json_path"] = json_path

    rows = []
    for cid, payload in sorted(cases.items()):
        ct = payload.get("ct_path")
        pet = payload.get("pet_path")
        js = payload.get("json_path")
        rows.append(
            {
                "case_id": cid,
                "ct_path": str(ct) if ct else "",
                "pet_path": str(pet) if pet else "",
                "json_path": str(js) if js else "",
                "has_ct": bool(ct),
                "has_pet": bool(pet),
                "has_json": bool(js),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    df = build_manifest(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logging.info("Saved manifest to %s with %d cases", args.output, len(df))


if __name__ == "__main__":
    main()
