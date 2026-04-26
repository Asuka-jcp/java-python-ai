from pathlib import Path

from src.label_utils import extract_suv_value, flatten_case_labels, safe_status_to_int


def test_status_parser_and_suv(tmp_path: Path):
    p = tmp_path / "CASE.json"
    p.write_text(
        '{"肺与胸膜": {"pet_status": "5", "suv": "SUVmax: 2.5", "ct_status": "3", "ct_description": "abc"}}',
        encoding="utf-8",
    )
    df = flatten_case_labels("CASE", p)
    assert int(df.loc[0, "pet_status"]) == 5
    assert int(df.loc[0, "ct_status"]) == 3
    assert float(df.loc[0, "suv"]) == 2.5


def test_safe_status_to_int_missing():
    assert safe_status_to_int("") == -1
    assert safe_status_to_int(None) == -1
    assert extract_suv_value("") != extract_suv_value("SUV=1.1")
