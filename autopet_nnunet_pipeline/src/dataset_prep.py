from __future__ import annotations

from typing import Any


def make_case_id(index: int) -> str:
    return f"AutoPET_{index:04d}"


def build_dataset_json(channel_0_name: str, channel_1_name: str, label_name: str, num_training: int) -> dict[str, Any]:
    return {
        "channel_names": {"0": channel_0_name, "1": channel_1_name},
        "labels": {"background": 0, label_name: 1},
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
