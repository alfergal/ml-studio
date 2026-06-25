import pandas as pd

from src.risk_segmentation import (
    DEFAULT_RISK_ACTIONS,
    add_risk_segments,
    assign_risk_segments,
    fit_risk_thresholds,
    summarize_risk_segments,
)


def test_fit_risk_thresholds_returns_expected_keys():
    probabilities = pd.Series([0.10, 0.20, 0.30, 0.40, 0.50])

    thresholds = fit_risk_thresholds(
        probabilities=probabilities,
        quantiles=(0.50, 0.75, 0.90),
    )

    assert set(thresholds.keys()) == {
        "low_medium",
        "medium_high",
        "high_critical",
    }

    assert thresholds["low_medium"] == 0.30


def test_assign_risk_segments_returns_ordered_segments():
    probabilities = pd.Series([0.10, 0.35, 0.65, 0.95])
    thresholds = {
        "low_medium": 0.25,
        "medium_high": 0.50,
        "high_critical": 0.80,
    }

    segments = assign_risk_segments(
        probabilities=probabilities,
        thresholds=thresholds,
    )

    assert list(segments.astype(str)) == [
        "low",
        "medium",
        "high",
        "critical",
    ]


def test_add_risk_segments_adds_probability_segment_target_and_action():
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
        }
    )

    probabilities = pd.Series([0.10, 0.35, 0.65, 0.95])
    y_true = pd.Series([0, 0, 1, 1])

    thresholds = {
        "low_medium": 0.25,
        "medium_high": 0.50,
        "high_critical": 0.80,
    }

    risk_scores = add_risk_segments(
        data=data,
        probabilities=probabilities,
        thresholds=thresholds,
        y_true=y_true,
    )

    assert "delay_probability" in risk_scores.columns
    assert "actual_delayed" in risk_scores.columns
    assert "risk_segment" in risk_scores.columns
    assert "recommended_action" in risk_scores.columns
    assert risk_scores.loc[0, "recommended_action"] == DEFAULT_RISK_ACTIONS["low"]


def test_summarize_risk_segments_returns_segment_summary():
    risk_scores = pd.DataFrame(
        {
            "delay_probability": [0.10, 0.20, 0.60, 0.90],
            "actual_delayed": [0, 0, 1, 1],
            "risk_segment": ["low", "low", "high", "critical"],
        }
    )

    summary = summarize_risk_segments(risk_scores)

    assert "cases" in summary.columns
    assert "actual_delay_rate" in summary.columns
    assert "delay_rate_lift_vs_average" in summary.columns
    assert "delayed_case_share" in summary.columns
    assert summary.loc["low", "cases"] == 2