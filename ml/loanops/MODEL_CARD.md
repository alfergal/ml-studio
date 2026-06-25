# LoanOps Delay Risk Model Card

## Model Overview

The LoanOps delay risk model predicts whether a loan application case is likely to experience an operational delay.

The project combines process mining features with machine learning to move from descriptive analysis toward early-warning operational risk monitoring.

The current modeling workflow includes:

- Process-enriched feature engineering.
- XGBoost delay prediction.
- Leakage-safe early modeling.
- Threshold optimization.
- Risk segmentation into low, medium, high and critical groups.
- Business-oriented recommended actions for each risk segment.

## Intended Use

The model is intended for portfolio and educational purposes as part of the ML Studio project.

From a business perspective, the model simulates how a banking operations team could prioritize loan applications according to predicted operational delay risk.

Potential use cases include:

- Early-warning monitoring.
- Operational risk prioritization.
- Manual review prioritization.
- Process bottleneck analysis.
- Decision-support for banking operations teams.

## Out-of-Scope Use

This model should not be used for real credit approval, customer eligibility decisions, pricing, underwriting or any decision that directly affects real customers.

The model predicts operational delay risk, not credit default risk or customer creditworthiness.

## Model Type

The main model is an XGBoost binary classifier.

The target is whether a loan application case becomes delayed.

The model outputs a probability, which is then converted into operational risk segments:

| Risk segment | Suggested action |
|---|---|
| Low | Standard automatic processing |
| Medium | Light monitoring |
| High | Priority review |
| Critical | Manual review and early-warning alert |

## Input Features

The model uses case-level process features derived from the event log.

Examples of feature categories include:

- Case duration features.
- Event count features.
- Activity frequency features.
- Activity presence indicators.
- Process mining derived features.
- Early-stage process features for leakage-safe modeling.

## Leakage-Safe Early Modeling

A key part of the project is separating retrospective modeling from early-warning modeling.

The retrospective model may use richer information available later in the process.

The leakage-safe early model only uses features that would be available early enough to support operational intervention.

This distinction is important because a model that uses future process information may look strong in evaluation but would not be valid for real early-warning use.

## Evaluation

The enriched modeling phase evaluates:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC AUC.
- Average precision.
- Threshold comparison.
- Risk segment delay rates.

The early model was further evaluated through risk segmentation.

Observed delay rate by segment:

| Risk segment | Actual delay rate |
|---|---:|
| Low | 8.4% |
| Medium | 37.2% |
| High | 43.7% |
| Critical | 49.4% |

The monotonic increase in actual delay rate shows that the model can rank cases by operational risk.

## Threshold Strategy

The project evaluates different classification thresholds.

The default threshold of 0.50 is not always optimal for early-warning systems because it may miss many delayed cases.

The pipeline therefore supports:

- Default threshold evaluation.
- Best-F1 threshold selection.
- Business threshold selection based on minimum precision.

This allows the model to be adapted to different operational strategies.

## Risk Segmentation

Instead of only returning a binary prediction, the model translates probabilities into risk segments.

This makes the output easier to use by non-technical business users.

The risk segmentation layer supports operational prioritization by mapping each case to a risk group and a recommended action.

## Limitations

This model is built for portfolio and educational purposes.

Current limitations include:

- The dataset is not a real production banking dataset.
- The model has not been validated in a real operational environment.
- The model does not account for regulatory, fairness or compliance requirements.
- The current pipeline is still evolving.
- The risk segments are useful for prioritization but should not be interpreted as final business decisions.
- The model should not be used for real customer-impacting decisions.

## Ethical Considerations

Operational risk models in banking contexts must be used carefully.

A real-world version would require:

- Bias and fairness analysis.
- Explainability review.
- Data governance.
- Monitoring for drift.
- Human oversight.
- Compliance validation.
- Clear separation between operational delay prediction and customer creditworthiness.

## Artifacts

Main artifacts produced by the project include:

- `notebooks/03_enriched_modeling.ipynb`
- `reports/early_risk_segment_summary.csv`
- `reports/early_risk_scores.csv`
- `reports/threshold_comparison.csv`
- `reports/feature_importance.csv`
- `models/xgboost_delay_model.joblib`
- `models/risk_segment_thresholds.json`

## Current Status

The project has moved from notebook-based analysis toward a reproducible ML pipeline.

The current focus is documenting the model, dataset, intended use and limitations clearly.