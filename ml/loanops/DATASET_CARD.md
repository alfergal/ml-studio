# LoanOps Dataset Card

## Dataset Overview

The LoanOps dataset represents a loan application process event log used for process mining and machine learning experimentation.

The dataset is used to build case-level features and predict operational delay risk in loan application workflows.

The project uses the dataset to simulate a banking operations use case where process events can be transformed into early-warning risk indicators.

## Purpose

The dataset supports the following tasks:

- Process mining analysis.
- Case-level feature engineering.
- Delay prediction.
- Early-warning modeling.
- Operational risk segmentation.
- Threshold optimization.
- Business-oriented monitoring reports.

## Data Structure

The source data is treated as an event log.

A typical event log contains:

- A case identifier.
- An activity name.
- A timestamp.
- Process-related attributes.
- Case outcome information.

The machine learning pipeline transforms the event log into a case-level feature table.

## Feature Engineering

The project derives features such as:

- Number of events per case.
- Number of unique activities per case.
- Case duration.
- Activity count features.
- Activity presence features.
- Process mining related indicators.
- Early-stage features for leakage-safe modeling.

These features are used to train binary classifiers for operational delay prediction.

## Target Variable

The target variable represents whether a loan application case experienced an operational delay.

The exact target column is configured in:

`config/model_config.yaml`

Default target name:

`is_delayed`

## Leakage Considerations

The project distinguishes between retrospective features and early-warning features.

Retrospective features may include information that is only available after the case has progressed or finished.

Early-warning features are designed to use only information available early enough to support intervention.

This separation is essential to avoid data leakage and to make the model more realistic for operational monitoring.

## Expected Processed Dataset

The reproducible pipeline expects a processed feature dataset at:

`data/processed/loanops_features.csv`

The path can be changed in:

`config/model_config.yaml`

Expected structure:

- One row per case.
- One target column.
- Numeric or encoded feature columns.
- Optional identifier columns such as `case_id`.

## Generated Outputs

The pipeline can generate:

- Model evaluation reports.
- Threshold comparison reports.
- Feature importance reports.
- Risk segmentation reports.
- Case-level risk scores.
- Saved model artifacts.
- Saved risk segmentation thresholds.

## Data Quality Considerations

Important checks for this dataset include:

- Missing timestamps.
- Duplicate events.
- Invalid case identifiers.
- Inconsistent activity names.
- Missing target values.
- Extreme case durations.
- Features that may introduce leakage.
- Train/test split consistency.

## Limitations

The dataset is used for portfolio and educational purposes.

Current limitations include:

- It should not be treated as a production banking dataset.
- It may not represent all real-world operational complexity.
- It may not include all compliance, fairness or customer impact variables required in real banking environments.
- Feature engineering choices are simplified for project purposes.
- The processed dataset must be generated consistently before running the training pipeline.

## Intended Use

The dataset is intended to support a portfolio-level ML project focused on process mining, operational delay prediction and risk segmentation.

It is suitable for demonstrating:

- Data preprocessing.
- Process mining feature engineering.
- Supervised machine learning.
- Model evaluation.
- Threshold tuning.
- Risk segmentation.
- Reproducible ML pipeline design.

## Not Intended For

The dataset should not be used for:

- Real credit approval.
- Real customer scoring.
- Real underwriting.
- Automated banking decisions.
- Regulatory production systems.
- Customer-impacting decisions.

## Current Status

The dataset documentation supports the transition from notebook-based experimentation to a more reproducible and maintainable ML project.