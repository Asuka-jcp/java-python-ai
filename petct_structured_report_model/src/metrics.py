from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_region_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regions: List[str],
    ignore_index: int = -1,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, region in enumerate(regions):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        mask = true_col != ignore_index
        if mask.sum() == 0:
            metrics[region] = {"accuracy": float("nan"), "macro_f1": float("nan"), "n": 0}
            continue
        t = true_col[mask]
        p = pred_col[mask]
        metrics[region] = {
            "accuracy": float(accuracy_score(t, p)),
            "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
            "n": int(mask.sum()),
        }
    return metrics


def summarize_region_metrics(metric_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    acc = [v["accuracy"] for v in metric_dict.values() if not np.isnan(v["accuracy"])]
    f1 = [v["macro_f1"] for v in metric_dict.values() if not np.isnan(v["macro_f1"])]
    n = [v["n"] for v in metric_dict.values()]
    return {
        "mean_accuracy": float(np.mean(acc)) if acc else float("nan"),
        "mean_macro_f1": float(np.mean(f1)) if f1 else float("nan"),
        "total_region_samples": int(np.sum(n)),
    }
