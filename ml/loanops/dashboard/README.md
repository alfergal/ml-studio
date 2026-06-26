# LoanOps Risk Dashboard

Streamlit dashboard for exploring the LoanOps operational delay risk segmentation results.

## Run locally

From `ml/loanops`:

    python -m streamlit run dashboard/app.py

## Input files

The dashboard reads the generated modeling reports from:

- `reports/early_risk_segment_summary.csv`
- `reports/early_risk_scores.csv`

## Main views

The dashboard includes:

- Total number of cases.
- Average delay rate.
- Critical segment delay rate.
- Actual delay rate by risk segment.
- Segment-level summary table.
- Business interpretation of each risk segment.
- Optional case-level risk score table.

## Purpose

The dashboard makes the LoanOps model outputs easier to inspect without opening notebooks.

It turns the risk segmentation layer into a visual business-facing tool for operational monitoring and review prioritization.
