from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    y_pred = np.asarray(probabilities) >= threshold

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def evaluate_threshold_grid(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    rows = [
        evaluate_binary_classifier(
            y_true=y_true,
            probabilities=probabilities,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(rows)


def find_best_f1_threshold(
    threshold_report: pd.DataFrame,
) -> pd.Series:
    return threshold_report.loc[
        threshold_report["f1"].idxmax()
    ]


def find_business_threshold(
    threshold_report: pd.DataFrame,
    min_precision: float = 0.50,
) -> pd.Series:
    candidates = threshold_report[
        threshold_report["precision"] >= min_precision
    ]

    if candidates.empty:
        return threshold_report.loc[
            threshold_report["precision"].idxmax()
        ]

    return candidates.loc[
        candidates["recall"].idxmax()
    ]


def save_report(
    report: pd.DataFrame,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(output_path, index=False)