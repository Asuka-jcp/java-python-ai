from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def load_nifti(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def normalize_ct(ct: np.ndarray, clip_min: float = -1000.0, clip_max: float = 1000.0) -> np.ndarray:
    ct = np.clip(ct, clip_min, clip_max)
    return (ct - clip_min) / (clip_max - clip_min + 1e-8)


def normalize_pet(pet: np.ndarray, clip_min: float = 0.0, clip_max: float = 20.0) -> np.ndarray:
    pet = np.clip(pet, clip_min, clip_max)
    return (pet - clip_min) / (clip_max - clip_min + 1e-8)


def center_crop_3d(array: np.ndarray, crop_shape: Sequence[int]) -> np.ndarray:
    z, y, x = array.shape
    cz, cy, cx = crop_shape
    sz = max((z - cz) // 2, 0)
    sy = max((y - cy) // 2, 0)
    sx = max((x - cx) // 2, 0)
    return array[sz : sz + cz, sy : sy + cy, sx : sx + cx]


def resize_3d(array: np.ndarray, out_shape: Sequence[int]) -> np.ndarray:
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=tuple(out_shape), mode="trilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def preprocess_ct_pet(
    ct: np.ndarray,
    pet: np.ndarray,
    resize_shape: Optional[Sequence[int]] = None,
    crop_shape: Optional[Sequence[int]] = None,
    ct_clip: Tuple[float, float] = (-1000.0, 1000.0),
    pet_clip: Tuple[float, float] = (0.0, 20.0),
) -> np.ndarray:
    if ct.shape != pet.shape:
        raise ValueError(f"CT/PET shape mismatch: {ct.shape} vs {pet.shape}")
    if crop_shape:
        ct = center_crop_3d(ct, crop_shape)
        pet = center_crop_3d(pet, crop_shape)
    if resize_shape:
        ct = resize_3d(ct, resize_shape)
        pet = resize_3d(pet, resize_shape)
    ct = normalize_ct(ct, clip_min=ct_clip[0], clip_max=ct_clip[1])
    pet = normalize_pet(pet, clip_min=pet_clip[0], clip_max=pet_clip[1])
    stacked = np.stack([ct, pet], axis=0)
    return stacked.astype(np.float32)
