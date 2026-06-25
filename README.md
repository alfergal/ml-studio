# ML Studio

Machine Learning portfolio focused on banking, process mining, predictive analytics, explainable AI and Bayesian modeling.

## Main Project: LoanOps Intelligence

LoanOps Intelligence is a banking-oriented machine learning project built on top of the BPI Challenge 2017 event log.

The goal is to analyze loan application processes, predict operational delay risk and explain the main drivers behind delayed cases.

## What the project does

- Transforms raw event logs into a case-level dataset.
- Performs exploratory analysis of banking process events.
- Engineers process-based features from loan applications.
- Trains classical machine learning models for delay prediction.
- Compares Random Forest and XGBoost.
- Evaluates models using Accuracy, ROC AUC and PR AUC.
- Applies SHAP for global and local explainability.
- Creates operational risk segments: low, medium, high and critical.
- Uses Bayesian logistic regression to estimate uncertainty around risk drivers.
- Computes 95% credible intervals for operational effects.

## Current results

| Model | Accuracy | ROC AUC | PR AUC |
|---|---:|---:|---:|
| Random Forest | 0.841 | 0.878 | 0.680 |
| XGBoost | 0.853 | 0.903 | 0.731 |

## Operational risk segmentation

| Segment | Cases | Average predicted delay risk | Actual delay rate |
|---|---:|---:|---:|
| Low | 4,128 | 0.063 | 0.058 |
| Medium | 648 | 0.361 | 0.386 |
| High | 989 | 0.636 | 0.653 |
| Critical | 537 | 0.803 | 0.819 |

## Bayesian interpretation

The Bayesian logistic regression model estimates the uncertainty of operational risk drivers.

| Feature | Mean effect | 95% credible interval | Interpretation |
|---|---:|---:|---|
| num_events | 1.594 | [1.340, 1.810] | Positive |
| num_validation_events | -2.392 | [-2.598, -2.183] | Negative |
| num_incomplete_file_calls | 0.153 | [-0.003, 0.327] | Uncertain |
| num_after_offer_calls | 0.128 | [0.045, 0.204] | Positive |
| requested_amount | 0.053 | [-0.020, 0.124] | Uncertain |

## Business interpretation

The model identifies delayed loan applications with strong predictive performance and provides explanations for operational risk.

The most relevant findings are:

- More process events strongly increase delay risk.
- After-offer calls increase delay risk.
- Validation events are associated with lower delay risk.
- Requested amount has weak evidence as a delay driver.
- Bayesian credible intervals help distinguish robust effects from uncertain ones.

## Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- PyMC
- ArviZ
- Matplotlib
- Seaborn
- PM4Py
- FastAPI
- Angular
- Docker

## Repository structure

```text
ml-studio/
├── backend/
├── docs/
├── frontend/
└── ml/
    └── loanops/
        ├── data/
        ├── models/
        ├── notebooks/
        ├── reports/
        ├── src/
        └── requirements.txt
```


## Next steps

- Refactor notebook logic into reusable Python modules.
- Add process mining visualizations with PM4Py.
- Build a FastAPI prediction service.
- Create an Angular dashboard for risk scoring.
- Add Docker support.
- Deploy the project as part of ML Studio.

## Enriched Modeling

This phase extends the original LoanOps process mining analysis with machine learning models for operational delay prediction.

The enriched modeling notebook includes:

* Process-enriched feature engineering.
* Baseline vs enriched model comparison.
* XGBoost delay prediction model.
* Leakage-safe early delay modeling.
* Threshold optimization for different operational strategies.
* Early risk segmentation into low, medium, high and critical groups.

### Main Result

The early risk segmentation layer converts model probabilities into actionable operational risk groups. The observed delay rate increases consistently across the predicted risk segments:

| Risk segment | Actual delay rate |
| ------------ | ----------------: |
| Low          |              8.4% |
| Medium       |             37.2% |
| High         |             43.7% |
| Critical     |             49.4% |

This confirms that the early model is not only useful for binary classification, but also for ranking loan applications by operational delay risk.

From a banking operations perspective, this enables prioritization of monitoring and manual review before delays fully materialize.

### Business Interpretation

The risk segments can support different operational actions:

| Risk segment | Suggested action                      |
| ------------ | ------------------------------------- |
| Low          | Standard automatic processing         |
| Medium       | Light monitoring                      |
| High         | Priority review                       |
| Critical     | Manual review and early-warning alert |

Overall, the enriched modeling phase turns the project from a descriptive process mining analysis into an early-warning risk monitoring system for loan application operations.

### Generated Outputs

This phase generates the following outputs:

* `ml/loanops/notebooks/03_enriched_modeling.ipynb`
* `ml/loanops/reports/early_risk_segment_summary.csv`
* `ml/loanops/reports/early_risk_scores.csv`
