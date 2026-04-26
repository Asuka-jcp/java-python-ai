#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import MultiRegionReportModel
from src.nifti_utils import load_nifti, preprocess_ct_pet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict structured report for one PET/CT case")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--pet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def invert_map(mp: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in mp.items()}


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ckpt = torch.load(args.model, map_location="cpu")

    regions = ckpt["region_names"]
    pet_map = {int(k): int(v) for k, v in ckpt["pet_map"].items()}
    ct_map = {int(k): int(v) for k, v in ckpt["ct_map"].items()}
    inv_pet = invert_map(pet_map)
    inv_ct = invert_map(ct_map)

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

    ct = load_nifti(args.ct)
    pet = load_nifti(args.pet)
    image = preprocess_ct_pet(
        ct,
        pet,
        resize_shape=cfg["training"].get("resize_shape"),
        crop_shape=cfg["training"].get("crop_shape"),
        ct_clip=tuple(cfg["training"]["normalize"]["ct_clip"]),
        pet_clip=tuple(cfg["training"]["normalize"]["pet_clip"]),
    )
    x = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        pet_pred = out["pet_logits"].argmax(dim=-1).squeeze(0).tolist()
        ct_pred = out["ct_logits"].argmax(dim=-1).squeeze(0).tolist()
        suv_pred = out.get("suv_pred")
        if suv_pred is not None:
            suv_pred = suv_pred.squeeze(0).tolist()

    report = {}
    for idx, region in enumerate(regions):
        report[region] = {
            "pet_status": str(inv_pet.get(int(pet_pred[idx]), "")),
            "ct_status": str(inv_ct.get(int(ct_pred[idx]), "")),
            "suv": "" if suv_pred is None else f"{float(suv_pred[idx]):.3f}",
            "ct_description": "",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
