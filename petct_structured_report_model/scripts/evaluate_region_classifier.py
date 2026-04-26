#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DatasetConfig, PETCTRegionDataset, collate_batch
from src.metrics import compute_region_classification_metrics, summarize_region_metrics
from src.model import MultiRegionReportModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PET/CT multi-region classifier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def invert_map(mp: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in mp.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest)
    labels = pd.read_csv(args.labels)

    ckpt = torch.load(args.model, map_location="cpu")
    regions = ckpt["region_names"]
    region_to_index = {r: i for i, r in enumerate(regions)}
    pet_map = {int(k): int(v) for k, v in ckpt["pet_map"].items()}
    ct_map = {int(k): int(v) for k, v in ckpt["ct_map"].items()}

    ds_cfg = DatasetConfig(
        resize_shape=cfg["training"].get("resize_shape"),
        crop_shape=cfg["training"].get("crop_shape"),
        ct_clip=cfg["training"]["normalize"]["ct_clip"],
        pet_clip=cfg["training"]["normalize"]["pet_clip"],
    )
    ds = PETCTRegionDataset(manifest, labels, region_to_index, pet_map, ct_map, ds_cfg)
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, collate_fn=collate_batch)

    model = MultiRegionReportModel(
        num_regions=len(regions),
        num_pet_classes=max(len(pet_map), 1),
        num_ct_classes=max(len(ct_map), 1),
        encoder_channels=tuple(cfg["model"]["encoder_channels"]),
        region_embed_dim=int(cfg["model"]["region_embed_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        use_suv_head=bool(cfg["model"].get("use_suv_head", True)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pet_true, pet_pred, ct_true, ct_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"])
            pet_true.append(batch["pet_targets"].numpy())
            ct_true.append(batch["ct_targets"].numpy())
            pet_pred.append(out["pet_logits"].argmax(dim=-1).cpu().numpy())
            ct_pred.append(out["ct_logits"].argmax(dim=-1).cpu().numpy())

    pet_true = np.concatenate(pet_true, axis=0)
    pet_pred = np.concatenate(pet_pred, axis=0)
    ct_true = np.concatenate(ct_true, axis=0)
    ct_pred = np.concatenate(ct_pred, axis=0)

    pet_metrics = compute_region_classification_metrics(pet_true, pet_pred, regions)
    ct_metrics = compute_region_classification_metrics(ct_true, ct_pred, regions)

    rows = []
    for r in regions:
        rows.append(
            {
                "region": r,
                "pet_accuracy": pet_metrics[r]["accuracy"],
                "pet_macro_f1": pet_metrics[r]["macro_f1"],
                "ct_accuracy": ct_metrics[r]["accuracy"],
                "ct_macro_f1": ct_metrics[r]["macro_f1"],
                "n": pet_metrics[r]["n"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "per_region_metrics.csv", index=False)
    summary = {
        "pet_summary": summarize_region_metrics(pet_metrics),
        "ct_summary": summarize_region_metrics(ct_metrics),
    }
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved evaluation outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
