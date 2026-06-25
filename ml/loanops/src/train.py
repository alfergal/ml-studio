from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify_values = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )


def build_xgboost_classifier(
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.90,
    colsample_bytree: float = 0.90,
    eval_metric: str = "logloss",
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric=eval_metric,
    )


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: XGBClassifier | None = None,
) -> XGBClassifier:
    if model is None:
        model = build_xgboost_classifier()

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_probabilities(
    model: XGBClassifier,
    X: pd.DataFrame,
) -> pd.Series:
    probabilities = model.predict_proba(X)[:, 1]

    return pd.Series(
        probabilities,
        index=X.index,
        name="delay_probability",
    )


def get_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def save_model(
    model: XGBClassifier,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        output_path,
    )


def load_model(
    path: str | Path,
) -> XGBClassifier:
    input_path = Path(path)

    return joblib.load(input_path)


def save_model_metadata(
    metadata: dict,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )