import numpy as np

from src.metrics import compute_region_classification_metrics, summarize_region_metrics


def test_metrics_compute():
    y_true = np.array([[0, 1], [1, -1]])
    y_pred = np.array([[0, 0], [1, 1]])
    regions = ["A", "B"]
    out = compute_region_classification_metrics(y_true, y_pred, regions)
    assert out["A"]["accuracy"] == 1.0
    assert out["B"]["n"] == 1
    summary = summarize_region_metrics(out)
    assert "mean_accuracy" in summary
