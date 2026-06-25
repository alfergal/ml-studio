import numpy as np
import pandas as pd

from src.evaluate import (
    evaluate_binary_classifier,
    evaluate_threshold_grid,
    find_best_f1_threshold,
    find_business_threshold,
)


def test_evaluate_binary_classifier_returns_expected_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.10, 0.40, 0.60, 0.90])

    metrics = evaluate_binary_classifier(
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.50,
    )

    assert metrics["threshold"] == 0.50
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert "roc_auc" in metrics
    assert "average_precision" in metrics


def test_evaluate_threshold_grid_returns_dataframe():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.10, 0.40, 0.60, 0.90])
    thresholds = np.array([0.30, 0.50, 0.70])

    report = evaluate_threshold_grid(
        y_true=y_true,
        probabilities=probabilities,
        thresholds=thresholds,
    )

    assert isinstance(report, pd.DataFrame)
    assert len(report) == 3
    assert set(["threshold", "accuracy", "precision", "recall", "f1"]).issubset(
        report.columns
    )


def test_find_best_f1_threshold_returns_highest_f1_row():
    report = pd.DataFrame(
        {
            "threshold": [0.30, 0.50, 0.70],
            "precision": [0.40, 0.70, 0.90],
            "recall": [0.90, 0.70, 0.30],
            "f1": [0.55, 0.70, 0.45],
        }
    )

    best = find_best_f1_threshold(report)

    assert best["threshold"] == 0.50
    assert best["f1"] == 0.70


def test_find_business_threshold_returns_highest_recall_with_min_precision():
    report = pd.DataFrame(
        {
            "threshold": [0.20, 0.40, 0.60],
            "precision": [0.30, 0.55, 0.80],
            "recall": [0.95, 0.70, 0.40],
            "f1": [0.45, 0.62, 0.53],
        }
    )

    threshold = find_business_threshold(
        threshold_report=report,
        min_precision=0.50,
    )

    assert threshold["threshold"] == 0.40