#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DatasetConfig, PETCTRegionDataset, collate_batch
from src.label_utils import MISSING_INT, build_status_maps
from src.model import MultiRegionReportModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PET/CT multi-region structured-report classifier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_for_task(logits: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    return criterion(logits.transpose(1, 2), target)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[(manifest["has_ct"] == True) & (manifest["has_pet"] == True)]
    labels = pd.read_csv(args.labels)

    region_names = sorted(labels["region"].dropna().unique().tolist())
    region_to_index = {r: i for i, r in enumerate(region_names)}
    status_maps = build_status_maps(labels, ["pet_status", "ct_status"])
    pet_map = status_maps["pet_status"]
    ct_map = status_maps["ct_status"]

    train_case, val_case = train_test_split(
        manifest["case_id"].tolist(),
        test_size=float(cfg["training"]["val_ratio"]),
        random_state=int(cfg.get("seed", 42)),
    )
    train_manifest = manifest[manifest["case_id"].isin(train_case)].reset_index(drop=True)
    val_manifest = manifest[manifest["case_id"].isin(val_case)].reset_index(drop=True)

    ds_cfg = DatasetConfig(
        resize_shape=cfg["training"].get("resize_shape"),
        crop_shape=cfg["training"].get("crop_shape"),
        ct_clip=cfg["training"]["normalize"]["ct_clip"],
        pet_clip=cfg["training"]["normalize"]["pet_clip"],
    )
    train_ds = PETCTRegionDataset(train_manifest, labels, region_to_index, pet_map, ct_map, ds_cfg)
    val_ds = PETCTRegionDataset(val_manifest, labels, region_to_index, pet_map, ct_map, ds_cfg)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"]["num_workers"]), collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"]["num_workers"]), collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiRegionReportModel(
        num_regions=len(region_names),
        num_pet_classes=max(len(pet_map), 1),
        num_ct_classes=max(len(ct_map), 1),
        encoder_channels=tuple(cfg["model"]["encoder_channels"]),
        region_embed_dim=int(cfg["model"]["region_embed_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        use_suv_head=bool(cfg["model"].get("use_suv_head", True)),
    ).to(device)

    ce_loss = nn.CrossEntropyLoss(ignore_index=MISSING_INT)
    mse_loss = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best_val = float("inf")

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train-{epoch}"):
            image = batch["image"].to(device)
            pet_t = batch["pet_targets"].to(device)
            ct_t = batch["ct_targets"].to(device)
            suv_t = batch["suv_targets"].to(device)

            out = model(image)
            loss = loss_for_task(out["pet_logits"], pet_t, ce_loss) + loss_for_task(out["ct_logits"], ct_t, ce_loss)
            if "suv_pred" in out:
                suv_pred = out["suv_pred"]
                mask = ~torch.isnan(suv_t)
                if mask.any():
                    loss = loss + float(cfg["loss"].get("suv_weight", 0.1)) * mse_loss(suv_pred[mask], suv_t[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val-{epoch}"):
                image = batch["image"].to(device)
                pet_t = batch["pet_targets"].to(device)
                ct_t = batch["ct_targets"].to(device)
                suv_t = batch["suv_targets"].to(device)
                out = model(image)
                loss = loss_for_task(out["pet_logits"], pet_t, ce_loss) + loss_for_task(out["ct_logits"], ct_t, ce_loss)
                if "suv_pred" in out:
                    mask = ~torch.isnan(suv_t)
                    if mask.any():
                        loss = loss + float(cfg["loss"].get("suv_weight", 0.1)) * mse_loss(out["suv_pred"][mask], suv_t[mask])
                val_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logging.info("Epoch %d train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "region_names": region_names,
                    "pet_map": pet_map,
                    "ct_map": ct_map,
                    "config": cfg,
                },
                args.output_dir / "best_model.pth",
            )

    pd.DataFrame(log_rows).to_csv(args.output_dir / "training_log.csv", index=False)
    (args.output_dir / "label_maps.json").write_text(
        json.dumps({"regions": region_names, "pet_map": pet_map, "ct_map": ct_map}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
