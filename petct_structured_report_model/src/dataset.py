from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .label_utils import MISSING_INT
from .nifti_utils import load_nifti, preprocess_ct_pet


@dataclass
class DatasetConfig:
    resize_shape: Optional[Sequence[int]] = None
    crop_shape: Optional[Sequence[int]] = None
    ct_clip: Sequence[float] = (-1000.0, 1000.0)
    pet_clip: Sequence[float] = (0.0, 20.0)


class PETCTRegionDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        region_to_index: Dict[str, int],
        pet_map: Dict[int, int],
        ct_map: Dict[int, int],
        config: DatasetConfig,
    ) -> None:
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.grouped = labels_df.groupby("case_id")
        self.region_to_index = region_to_index
        self.pet_map = pet_map
        self.ct_map = ct_map
        self.config = config
        self.regions = [r for r, _ in sorted(region_to_index.items(), key=lambda x: x[1])]

    def __len__(self) -> int:
        return len(self.manifest_df)

    def _build_targets(self, case_id: str) -> Dict[str, torch.Tensor]:
        n_regions = len(self.region_to_index)
        pet_targets = np.full(n_regions, MISSING_INT, dtype=np.int64)
        ct_targets = np.full(n_regions, MISSING_INT, dtype=np.int64)
        suv_targets = np.full(n_regions, np.nan, dtype=np.float32)

        if case_id in self.grouped.groups:
            case_df = self.grouped.get_group(case_id)
            for _, row in case_df.iterrows():
                region = row["region"]
                if region not in self.region_to_index:
                    continue
                idx = self.region_to_index[region]
                pet_raw = int(row["pet_status"])
                ct_raw = int(row["ct_status"])
                pet_targets[idx] = self.pet_map.get(pet_raw, MISSING_INT)
                ct_targets[idx] = self.ct_map.get(ct_raw, MISSING_INT)
                suv_targets[idx] = float(row["suv"])

        return {
            "pet_targets": torch.tensor(pet_targets, dtype=torch.long),
            "ct_targets": torch.tensor(ct_targets, dtype=torch.long),
            "suv_targets": torch.tensor(suv_targets, dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.manifest_df.iloc[index]
        case_id = row["case_id"]
        ct = load_nifti(Path(row["ct_path"]))
        pet = load_nifti(Path(row["pet_path"]))
        image = preprocess_ct_pet(
            ct,
            pet,
            resize_shape=self.config.resize_shape,
            crop_shape=self.config.crop_shape,
            ct_clip=tuple(self.config.ct_clip),
            pet_clip=tuple(self.config.pet_clip),
        )
        targets = self._build_targets(case_id)
        return {
            "case_id": case_id,
            "image": torch.tensor(image, dtype=torch.float32),
            **targets,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "case_id": [x["case_id"] for x in batch],
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "pet_targets": torch.stack([x["pet_targets"] for x in batch], dim=0),
        "ct_targets": torch.stack([x["ct_targets"] for x in batch], dim=0),
        "suv_targets": torch.stack([x["suv_targets"] for x in batch], dim=0),
    }
