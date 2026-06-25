# LoanOps Risk Service API

FastAPI service for operational delay risk prediction and risk segmentation.

## Run locally

From `ml/loanops`:

    python -m uvicorn api.main:app --reload

API documentation:

    http://127.0.0.1:8000/docs

## Endpoints

### Health check

    GET /health

Returns:

    {
      "status": "ok"
    }

### Predict delay risk

    POST /predict

Request body:

    {
      "features": {
        "feature_a": 1.0,
        "feature_b": 2.0
      }
    }

Response body:

    {
      "delay_probability": 0.42,
      "risk_segment": "high"
    }

The API expects a trained model at:

    models/xgboost_delay_model.joblib

and risk segmentation thresholds at:

    models/risk_segment_thresholds.json
