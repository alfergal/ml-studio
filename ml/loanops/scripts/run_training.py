from argparse import ArgumentParser
from src.config import load_config
import pandas as pd

from src.evaluate import (
    evaluate_binary_classifier,
    evaluate_threshold_grid,
    find_best_f1_threshold,
    find_business_threshold,
    save_report,
)
from src.features import build_feature_target_split, load_event_log
from src.risk_segmentation import (
    add_risk_segments,
    fit_risk_thresholds,
    save_risk_thresholds,
    summarize_risk_segments,
)
from src.train import (
    build_xgboost_classifier,
    get_feature_importance,
    predict_probabilities,
    save_model,
    save_model_metadata,
    split_train_test,
    train_classifier,
)

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/model_config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    data = load_event_log(config["data"]["features_path"])

    X, y = build_feature_target_split(
        data=data,
        target_col=config["data"]["target_column"],
        drop_cols=config["data"].get("drop_columns", []),
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X=X,
        y=y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=config["split"]["stratify"],
    )

    model = build_xgboost_classifier(
        random_state=config["model"]["random_state"],
        n_estimators=config["model"]["n_estimators"],
        learning_rate=config["model"]["learning_rate"],
        max_depth=config["model"]["max_depth"],
        subsample=config["model"]["subsample"],
        colsample_bytree=config["model"]["colsample_bytree"],
        eval_metric=config["model"]["eval_metric"],
    )

    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        model=model,
    )

    test_probabilities = predict_probabilities(
        model=model,
        X=X_test,
    )

    threshold_report = evaluate_threshold_grid(
        y_true=y_test,
        probabilities=test_probabilities,
    )

    best_f1_threshold = find_best_f1_threshold(
        threshold_report=threshold_report,
    )

    business_threshold = find_business_threshold(
        threshold_report=threshold_report,
        min_precision=config["thresholds"]["min_business_precision"],
    )

    default_metrics = evaluate_binary_classifier(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=config["thresholds"]["default_threshold"],
    )

    feature_importance = get_feature_importance(
        model=model,
        feature_names=list(X_train.columns),
    )

    risk_thresholds = fit_risk_thresholds(
        probabilities=test_probabilities,
        quantiles=tuple(config["risk_segmentation"]["quantiles"]),
    )

    risk_scores = add_risk_segments(
        data=X_test,
        probabilities=test_probabilities,
        thresholds=risk_thresholds,
        y_true=y_test,
    )

    risk_segment_summary = summarize_risk_segments(
        risk_scores=risk_scores,
    )

    save_model(
        model=model,
        path=config["outputs"]["model_path"],
    )

    save_risk_thresholds(
        thresholds=risk_thresholds,
        path=config["outputs"]["risk_thresholds_path"],
    )

    save_report(
        report=threshold_report,
        path=config["outputs"]["threshold_report_path"],
    )

    save_report(
        report=feature_importance,
        path=config["outputs"]["feature_importance_path"],
    )

    save_report(
        report=risk_segment_summary.reset_index(),
        path=config["outputs"]["risk_segment_summary_path"],
    )

    save_report(
        report=risk_scores,
        path=config["outputs"]["risk_scores_path"],
    )

    metadata = {
        "default_metrics": default_metrics,
        "best_f1_threshold": best_f1_threshold.to_dict(),
        "business_threshold": business_threshold.to_dict(),
        "features_path": config["data"]["features_path"],
        "target_column": config["data"]["target_column"],
    }

    save_model_metadata(
        metadata=metadata,
        path=config["outputs"]["model_metadata_path"],
    )


if __name__ == "__main__":
    main()