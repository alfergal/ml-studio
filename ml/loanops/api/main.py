from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.risk_segmentation import (
    assign_risk_segments,
    load_risk_thresholds,
)
from src.train import load_model


MODEL_PATH = Path("models/xgboost_delay_model.joblib")
RISK_THRESHOLDS_PATH = Path("models/risk_segment_thresholds.json")

app = FastAPI(
    title="LoanOps Risk Service",
    description="API for operational delay risk prediction and risk segmentation.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, Any]


class PredictionResponse(BaseModel):
    delay_probability: float
    risk_segment: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_delay_risk(request: PredictionRequest) -> PredictionResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {MODEL_PATH}",
        )

    if not RISK_THRESHOLDS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Risk thresholds file not found: {RISK_THRESHOLDS_PATH}",
        )

    model = load_model(MODEL_PATH)
    risk_thresholds = load_risk_thresholds(RISK_THRESHOLDS_PATH)

    input_data = pd.DataFrame([request.features])

    probability = float(model.predict_proba(input_data)[:, 1][0])

    risk_segment = assign_risk_segments(
        probabilities=pd.Series([probability]),
        thresholds=risk_thresholds,
    ).astype(str).iloc[0]

    return PredictionResponse(
        delay_probability=probability,
        risk_segment=risk_segment,
    )