from pathlib import Path
import json

import numpy as np
import pandas as pd


RISK_SEGMENT_ORDER = ["low", "medium", "high", "critical"]

DEFAULT_RISK_ACTIONS = {
    "low": "standard automatic processing",
    "medium": "light monitoring",
    "high": "priority review",
    "critical": "manual review and early warning alert",
}


def fit_risk_thresholds(
    probabilities: pd.Series | np.ndarray,
    quantiles: tuple[float, float, float] = (0.50, 0.75, 0.90),
) -> dict[str, float]:
    probability_series = pd.Series(probabilities, dtype=float)

    thresholds = probability_series.quantile(quantiles)

    return {
        "low_medium": float(thresholds.iloc[0]),
        "medium_high": float(thresholds.iloc[1]),
        "high_critical": float(thresholds.iloc[2]),
    }


def assign_risk_segments(
    probabilities: pd.Series | np.ndarray,
    thresholds: dict[str, float],
) -> pd.Series:
    probability_series = pd.Series(probabilities, dtype=float)

    conditions = [
        probability_series <= thresholds["low_medium"],
        probability_series <= thresholds["medium_high"],
        probability_series <= thresholds["high_critical"],
    ]

    segments = np.select(
        conditions,
        ["low", "medium", "high"],
        default="critical",
    )

    return pd.Series(
        pd.Categorical(
            segments,
            categories=RISK_SEGMENT_ORDER,
            ordered=True,
        ),
        index=probability_series.index,
        name="risk_segment",
    )


def add_risk_segments(
    data: pd.DataFrame,
    probabilities: pd.Series | np.ndarray,
    thresholds: dict[str, float],
    y_true: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    risk_scores = data.copy()

    risk_scores["delay_probability"] = pd.Series(
        probabilities,
        index=risk_scores.index,
        dtype=float,
    )

    if y_true is not None:
        risk_scores["actual_delayed"] = pd.Series(
            y_true,
            index=risk_scores.index,
        ).values

    risk_scores["risk_segment"] = assign_risk_segments(
        risk_scores["delay_probability"],
        thresholds,
    ).values

    risk_scores["recommended_action"] = (
        risk_scores["risk_segment"]
        .astype(str)
        .map(DEFAULT_RISK_ACTIONS)
    )

    return risk_scores


def summarize_risk_segments(
    risk_scores: pd.DataFrame,
    probability_col: str = "delay_probability",
    target_col: str = "actual_delayed",
    segment_col: str = "risk_segment",
) -> pd.DataFrame:
    summary = (
        risk_scores
        .groupby(segment_col, observed=True)
        .agg(
            cases=(probability_col, "count"),
            avg_delay_probability=(probability_col, "mean"),
            actual_delay_rate=(target_col, "mean"),
            min_probability=(probability_col, "min"),
            max_probability=(probability_col, "max"),
        )
        .sort_index()
    )

    overall_delay_rate = risk_scores[target_col].mean()

    summary["delay_rate_lift_vs_average"] = (
        summary["actual_delay_rate"] / overall_delay_rate
    )

    summary["delayed_cases"] = (
        summary["cases"] * summary["actual_delay_rate"]
    ).round().astype(int)

    summary["delayed_case_share"] = (
        summary["delayed_cases"] / risk_scores[target_col].sum()
    )

    return summary


def save_risk_thresholds(
    thresholds: dict[str, float],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=4)


def load_risk_thresholds(path: str | Path) -> dict[str, float]:
    input_path = Path(path)

    with input_path.open("r", encoding="utf-8") as file:
        thresholds = json.load(file)

    return {key: float(value) for key, value in thresholds.items()}