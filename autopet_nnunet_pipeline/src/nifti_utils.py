from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def _validate_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NIfTI file not found: {p}")
    return p


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    p = _validate_path(path)
    return nib.load(str(p))


def get_spacing(path: str | Path) -> Tuple[float, float, float]:
    img = load_nifti(path)
    return tuple(float(x) for x in img.header.get_zooms()[:3])


def get_shape(path: str | Path) -> Tuple[int, ...]:
    img = load_nifti(path)
    return tuple(int(x) for x in img.shape)


def get_affine(path: str | Path) -> np.ndarray:
    img = load_nifti(path)
    return np.asarray(img.affine, dtype=np.float64)


def check_binary_label(path: str | Path) -> bool:
    img = load_nifti(path)
    data = np.asarray(img.get_fdata())
    unique_values = np.unique(data)
    return set(unique_values.tolist()).issubset({0.0, 1.0})
