from pathlib import Path

import pandas as pd


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def load_event_log(
    path: str | Path,
) -> pd.DataFrame:
    input_path = Path(path)

    if input_path.suffix == ".csv":
        return pd.read_csv(input_path)

    if input_path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(input_path)

    raise ValueError(
        f"Unsupported file format: {input_path.suffix}"
    )


def parse_timestamp_column(
    data: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    output = data.copy()

    validate_required_columns(
        output,
        [timestamp_col],
    )

    output[timestamp_col] = pd.to_datetime(
        output[timestamp_col],
        errors="coerce",
    )

    return output


def build_case_level_features(
    event_log: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str | None = None,
) -> pd.DataFrame:
    required_columns = [case_id_col, activity_col]

    if timestamp_col is not None:
        required_columns.append(timestamp_col)

    validate_required_columns(
        event_log,
        required_columns,
    )

    data = event_log.copy()

    if timestamp_col is not None:
        data = parse_timestamp_column(
            data,
            timestamp_col,
        )

    features = (
        data
        .groupby(case_id_col)
        .agg(
            event_count=(activity_col, "count"),
            unique_activity_count=(activity_col, "nunique"),
        )
    )

    if timestamp_col is not None:
        time_features = (
            data
            .groupby(case_id_col)
            .agg(
                case_start_time=(timestamp_col, "min"),
                case_end_time=(timestamp_col, "max"),
            )
        )

        time_features["case_duration_hours"] = (
            time_features["case_end_time"] - time_features["case_start_time"]
        ).dt.total_seconds() / 3600

        features = features.join(time_features)

    return features.reset_index()


def add_activity_count_features(
    features: pd.DataFrame,
    event_log: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    prefix: str = "activity_count",
) -> pd.DataFrame:
    validate_required_columns(
        features,
        [case_id_col],
    )

    validate_required_columns(
        event_log,
        [case_id_col, activity_col],
    )

    activity_counts = (
        event_log
        .groupby([case_id_col, activity_col])
        .size()
        .unstack(fill_value=0)
    )

    activity_counts.columns = [
        f"{prefix}_{str(column).lower().replace(' ', '_')}"
        for column in activity_counts.columns
    ]

    return features.merge(
        activity_counts.reset_index(),
        on=case_id_col,
        how="left",
    ).fillna(0)


def add_activity_presence_features(
    features: pd.DataFrame,
    event_log: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    prefix: str = "has_activity",
) -> pd.DataFrame:
    output = add_activity_count_features(
        features=features,
        event_log=event_log,
        case_id_col=case_id_col,
        activity_col=activity_col,
        prefix=prefix,
    )

    activity_columns = [
        column for column in output.columns
        if column.startswith(f"{prefix}_")
    ]

    output[activity_columns] = (
        output[activity_columns] > 0
    ).astype(int)

    return output


def build_feature_target_split(
    data: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    validate_required_columns(
        data,
        [target_col],
    )

    if drop_cols is None:
        drop_cols = []

    columns_to_drop = [
        column for column in [target_col, *drop_cols]
        if column in data.columns
    ]

    X = data.drop(columns=columns_to_drop)
    y = data[target_col]

    return X, y


def save_features(
    features: pd.DataFrame,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        features.to_csv(output_path, index=False)
        return

    if output_path.suffix in [".parquet", ".pq"]:
        features.to_parquet(output_path, index=False)
        return

    raise ValueError(
        f"Unsupported file format: {output_path.suffix}"
    )