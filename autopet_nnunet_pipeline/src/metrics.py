from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, label: np.ndarray) -> float:
    """Compute Dice score for binary arrays."""
    pred_bin = (pred > 0).astype(np.uint8)
    label_bin = (label > 0).astype(np.uint8)

    intersection = np.sum(pred_bin * label_bin)
    pred_sum = np.sum(pred_bin)
    label_sum = np.sum(label_bin)

    if pred_sum + label_sum == 0:
        return 1.0

    return float(2.0 * intersection / (pred_sum + label_sum))


def false_positive_volume(pred: np.ndarray, label: np.ndarray, voxel_volume: float = 1.0) -> float:
    """Compute false positive volume given voxel volume (mm^3)."""
    pred_bin = (pred > 0).astype(np.uint8)
    label_bin = (label > 0).astype(np.uint8)
    fp_voxels = np.sum((pred_bin == 1) & (label_bin == 0))
    return float(fp_voxels * voxel_volume)


def false_negative_volume(pred: np.ndarray, label: np.ndarray, voxel_volume: float = 1.0) -> float:
    """Compute false negative volume given voxel volume (mm^3)."""
    pred_bin = (pred > 0).astype(np.uint8)
    label_bin = (label > 0).astype(np.uint8)
    fn_voxels = np.sum((pred_bin == 0) & (label_bin == 1))
    return float(fn_voxels * voxel_volume)
