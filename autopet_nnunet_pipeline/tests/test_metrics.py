import numpy as np

from src.metrics import dice_score, false_negative_volume, false_positive_volume


def test_dice_score() -> None:
    pred = np.array([[1, 1, 0, 0]])
    label = np.array([[1, 0, 1, 0]])
    # intersection=1, pred_sum=2, label_sum=2 => dice=0.5
    assert dice_score(pred, label) == 0.5


def test_false_positive_volume() -> None:
    pred = np.array([[1, 1, 0, 0]])
    label = np.array([[1, 0, 1, 0]])
    # one FP voxel
    assert false_positive_volume(pred, label, voxel_volume=2.0) == 2.0


def test_false_negative_volume() -> None:
    pred = np.array([[1, 1, 0, 0]])
    label = np.array([[1, 0, 1, 0]])
    # one FN voxel
    assert false_negative_volume(pred, label, voxel_volume=3.0) == 3.0
