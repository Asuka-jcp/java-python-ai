from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

MISSING_INT = -1
SUV_REGEX = re.compile(r"[-+]?\d*\.?\d+")


def parse_json_label(json_path: Path) -> Dict[str, Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be dict, got {type(data)} in {json_path}")
    return data


def safe_status_to_int(value: Any, missing_value: int = MISSING_INT) -> int:
    if value is None:
        return missing_value
    text = str(value).strip()
    if not text:
        return missing_value
    try:
        return int(float(text))
    except ValueError:
        return missing_value


def extract_suv_value(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    match = SUV_REGEX.search(text)
    if not match:
        return float("nan")
    return float(match.group(0))


def flatten_case_labels(case_id: str, json_path: Path) -> pd.DataFrame:
    data = parse_json_label(json_path)
    rows = []
    for region, payload in data.items():
        payload = payload or {}
        rows.append(
            {
                "case_id": case_id,
                "region": region,
                "pet_status": safe_status_to_int(payload.get("pet_status")),
                "suv": extract_suv_value(payload.get("suv")),
                "ct_status": safe_status_to_int(payload.get("ct_status")),
                "ct_description": str(payload.get("ct_description", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def build_status_maps(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, Dict[int, int]]:
    maps: Dict[str, Dict[int, int]] = {}
    for col in columns:
        values = sorted(v for v in df[col].dropna().astype(int).unique().tolist() if v != MISSING_INT)
        maps[col] = {int(v): idx for idx, v in enumerate(values)}
    return maps


def encode_status_columns(
    df: pd.DataFrame,
    status_maps: Dict[str, Dict[int, int]],
    missing_value: int = MISSING_INT,
) -> pd.DataFrame:
    out = df.copy()
    for col, mapper in status_maps.items():
        out[f"{col}_encoded"] = [mapper.get(int(v), missing_value) if pd.notna(v) else missing_value for v in out[col]]
    return out


def summarize_distributions(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    region_dist = df["region"].value_counts(dropna=False)
    pet_dist = df["pet_status"].value_counts(dropna=False).sort_index()
    ct_dist = df["ct_status"].value_counts(dropna=False).sort_index()
    return region_dist, pet_dist, ct_dist


def is_missing_label(value: Optional[int]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return int(value) == MISSING_INT
