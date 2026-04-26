from pathlib import Path

from scripts.inspect_dataset import build_manifest


def test_build_manifest(tmp_path: Path):
    (tmp_path / "CASE_0000.nii.gz").write_text("x")
    (tmp_path / "CASE_0001.nii.gz").write_text("x")
    (tmp_path / "CASE.json").write_text("{}")

    df = build_manifest(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["case_id"] == "CASE"
    assert bool(row["has_ct"]) and bool(row["has_pet"]) and bool(row["has_json"])
