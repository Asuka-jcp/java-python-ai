import json
from pathlib import Path

from src.dataset_prep import build_dataset_json, make_case_id


def test_dataset_json_schema() -> None:
    data = build_dataset_json("CT", "SUV", "lesion", 5)
    assert data["channel_names"] == {"0": "CT", "1": "SUV"}
    assert data["labels"] == {"background": 0, "lesion": 1}
    assert data["numTraining"] == 5
    assert data["file_ending"] == ".nii.gz"


def test_nnunet_filename_rules(tmp_path: Path) -> None:
    case_id = make_case_id(1)
    image0 = tmp_path / f"{case_id}_0000.nii.gz"
    image1 = tmp_path / f"{case_id}_0001.nii.gz"
    label = tmp_path / f"{case_id}.nii.gz"

    image0.write_bytes(b"dummy")
    image1.write_bytes(b"dummy")
    label.write_bytes(b"dummy")

    assert image0.name == "AutoPET_0001_0000.nii.gz"
    assert image1.name == "AutoPET_0001_0001.nii.gz"
    assert label.name == "AutoPET_0001.nii.gz"


def test_dataset_json_serializable() -> None:
    data = build_dataset_json("CT", "SUV", "lesion", 12)
    dumped = json.dumps(data)
    loaded = json.loads(dumped)
    assert loaded["numTraining"] == 12
