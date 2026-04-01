from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from .config import ProjectConfig
from .data import (
    load_price_index,
    load_processed_dataset,
    load_raw_transactions,
    save_processed_dataset,
)
from .features import build_feature_dataset
from .modeling import compute_regression_metrics, fit_quantile_models, predict_with_intervals
from .modeling import split_dataset_by_time
from .reporting import plot_prediction_interval_sample, write_project_summary


def prepare_dataset(config: ProjectConfig) -> tuple[pd.DataFrame, list[str]]:
    paths = config.paths
    paths.ensure_directories()

    transactions = load_raw_transactions(paths)
    price_index = load_price_index(paths)
    dataset, feature_columns = build_feature_dataset(transactions, price_index, config.features)
    save_processed_dataset(dataset, feature_columns, paths)

    return dataset, feature_columns


def load_or_prepare_dataset(config: ProjectConfig) -> tuple[pd.DataFrame, list[str]]:
    if config.paths.processed_dataset.exists() and config.paths.features_path.exists():
        return load_processed_dataset(config.paths)
    return prepare_dataset(config)


def train_pipeline(
    config: ProjectConfig,
    sample_rows: int | None = None,
) -> dict[str, dict[str, float]]:
    paths = config.paths
    paths.ensure_directories()

    dataset, feature_columns = load_or_prepare_dataset(config)
    split_frames = split_dataset_by_time(dataset, config.split)

    if sample_rows is not None:
        split_frames = {
            name: frame.sample(n=min(sample_rows, len(frame)), random_state=config.model.random_state)
            .sort_values("month")
            .reset_index(drop=True)
            for name, frame in split_frames.items()
        }

    if split_frames["train"].empty or split_frames["validation"].empty or split_frames["test"].empty:
        raise ValueError(
            "One of the dataset splits is empty. Adjust the split boundaries in ProjectConfig."
        )

    models = fit_quantile_models(
        X_train=split_frames["train"],
        y_train=split_frames["train"]["resale_price"],
        feature_columns=feature_columns,
        categorical_features=config.features.categorical_features,
        config=config.model,
    )

    metrics_by_split: dict[str, dict[str, float]] = {}
    prediction_frames: list[pd.DataFrame] = []

    for split_name in ("validation", "test"):
        frame = split_frames[split_name].copy()
        predictions = predict_with_intervals(models, frame, feature_columns)
        metrics_by_split[split_name] = compute_regression_metrics(
            y_true=frame["resale_price"],
            lower=predictions["lower"],
            median=predictions["median"],
            upper=predictions["upper"],
        )

        prediction_frame = frame[
            ["month", "town", "flat_type", "storey_range", "floor_area_sqm", "resale_price"]
        ].copy()
        prediction_frame["split"] = split_name
        prediction_frame = prediction_frame.rename(columns={"resale_price": "actual_price"})
        prediction_frame["lower_prediction"] = predictions["lower"]
        prediction_frame["median_prediction"] = predictions["median"]
        prediction_frame["upper_prediction"] = predictions["upper"]
        prediction_frame["absolute_error"] = (
            prediction_frame["actual_price"] - prediction_frame["median_prediction"]
        ).abs()
        prediction_frames.append(prediction_frame)

        figure_path = paths.figures_dir / f"{split_name}_prediction_intervals.png"
        plot_prediction_interval_sample(prediction_frame, split_name, figure_path)

    predictions_output = pd.concat(prediction_frames, ignore_index=True)
    predictions_output.to_csv(paths.predictions_path, index=False)

    model_files: dict[str, str] = {}
    for quantile, model in models.items():
        suffix = int(round(quantile * 100))
        model_path = paths.models_dir / f"quantile_{suffix:02d}.joblib"
        joblib.dump(model, model_path)
        model_files[f"q{suffix:02d}"] = str(model_path)

    split_sizes = {name: int(len(frame)) for name, frame in split_frames.items()}
    metrics_bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feature_columns),
        "sample_rows": sample_rows,
        "split_sizes": split_sizes,
        "metrics": metrics_by_split,
    }
    paths.metrics_path.write_text(json.dumps(metrics_bundle, indent=2))

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_serialisable_dict(),
        "features": feature_columns,
        "sample_rows": sample_rows,
        "model_files": model_files,
        "split_sizes": split_sizes,
    }
    paths.model_metadata_path.write_text(json.dumps(metadata, indent=2))

    write_project_summary(
        metrics_by_split=metrics_by_split,
        split_sizes=split_sizes,
        feature_count=len(feature_columns),
        output_path=paths.summary_path,
    )

    return metrics_by_split
