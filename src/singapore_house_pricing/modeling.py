from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import ModelConfig, SplitConfig


def split_dataset_by_time(
    dataframe: pd.DataFrame,
    split_config: SplitConfig,
) -> dict[str, pd.DataFrame]:
    train_end = pd.Timestamp(split_config.train_end)
    validation_end = pd.Timestamp(split_config.validation_end)

    train_frame = dataframe.loc[dataframe["month"] < train_end].copy()
    validation_frame = dataframe.loc[
        (dataframe["month"] >= train_end) & (dataframe["month"] < validation_end)
    ].copy()
    test_frame = dataframe.loc[dataframe["month"] >= validation_end].copy()

    return {
        "train": train_frame.reset_index(drop=True),
        "validation": validation_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
    }


def build_quantile_pipeline(
    feature_columns: list[str],
    categorical_features: tuple[str, ...],
    quantile: float,
    config: ModelConfig,
) -> Pipeline:
    categorical_columns = [column for column in feature_columns if column in categorical_features]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
        ]
    )

    regressor = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        subsample=config.subsample,
        verbose=1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    categorical_features: tuple[str, ...],
    config: ModelConfig,
) -> dict[float, Pipeline]:
    models: dict[float, Pipeline] = {}
    for quantile in config.quantiles:
        pipeline = build_quantile_pipeline(
            feature_columns=feature_columns,
            categorical_features=categorical_features,
            quantile=quantile,
            config=config,
        )
        pipeline.fit(X_train[feature_columns], y_train)
        models[quantile] = pipeline
    return models


def enforce_interval_order(
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.sort(np.column_stack([lower, median, upper]), axis=1)
    return ordered[:, 0], ordered[:, 1], ordered[:, 2]


def predict_with_intervals(
    models: dict[float, Pipeline],
    features: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, np.ndarray]:
    predictions = {
        quantile: model.predict(features[feature_columns]) for quantile, model in models.items()
    }

    ordered_quantiles = sorted(predictions)
    lower = predictions[ordered_quantiles[0]]
    median = predictions[ordered_quantiles[1]]
    upper = predictions[ordered_quantiles[2]]
    lower, median, upper = enforce_interval_order(lower, median, upper)
    return {"lower": lower, "median": median, "upper": upper}


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    lower, median, upper = enforce_interval_order(lower, median, upper)

    safe_denominator = np.where(actual == 0.0, np.nan, actual)

    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, median))),
        "mae": float(mean_absolute_error(actual, median)),
        "mape": float(np.nanmean(np.abs((actual - median) / safe_denominator)) * 100),
        "mdape": float(np.nanmedian(np.abs((actual - median) / safe_denominator)) * 100),
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
        "pinball_loss_p10": float(mean_pinball_loss(actual, lower, alpha=0.1)),
        "pinball_loss_p50": float(mean_pinball_loss(actual, median, alpha=0.5)),
        "pinball_loss_p90": float(mean_pinball_loss(actual, upper, alpha=0.9)),
    }
