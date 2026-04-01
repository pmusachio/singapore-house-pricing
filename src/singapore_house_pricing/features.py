from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import FeatureConfig


BASE_NUMERIC_FEATURES = (
    "floor_area_sqm",
    "lease_commence_date",
    "remaining_lease_years",
    "lease_age",
    "month_num",
    "quarter",
    "transaction_year",
    "storey_midpoint",
    "flat_type_room_count",
    "price_index",
    "inflation_adjustment_factor",
)


def quarter_to_timestamp(label: str) -> pd.Timestamp:
    match = re.fullmatch(r"(\d{4})-Q([1-4])", str(label).strip())
    if match is None:
        raise ValueError(f"Unsupported quarter label: {label}")
    year, quarter = (int(part) for part in match.groups())
    month = ((quarter - 1) * 3) + 1
    return pd.Timestamp(year=year, month=month, day=1)


def month_to_quarter(value: pd.Timestamp) -> str:
    return f"{value.year}-Q{((value.month - 1) // 3) + 1}"


def parse_remaining_lease(lease_text: str | float | None) -> float:
    if lease_text is None or (isinstance(lease_text, float) and np.isnan(lease_text)):
        return np.nan

    text = str(lease_text).strip().lower()
    if not text:
        return np.nan

    years_match = re.search(r"(\d+)\s+year", text)
    months_match = re.search(r"(\d+)\s+month", text)

    years = float(years_match.group(1)) if years_match else 0.0
    months = float(months_match.group(1)) if months_match else 0.0

    if years == 0.0 and months == 0.0:
        return np.nan

    return years + (months / 12.0)


def parse_storey_midpoint(storey_range: str | float | None) -> float:
    if storey_range is None or (isinstance(storey_range, float) and np.isnan(storey_range)):
        return np.nan

    numbers = [float(number) for number in re.findall(r"\d+", str(storey_range))]
    if not numbers:
        return np.nan

    return float(sum(numbers) / len(numbers))


def parse_flat_type_room_count(flat_type: str | float | None) -> float:
    if flat_type is None or (isinstance(flat_type, float) and np.isnan(flat_type)):
        return np.nan

    match = re.match(r"(\d+)", str(flat_type).strip())
    if match is None:
        return np.nan

    return float(match.group(1))


def attach_market_index(
    dataframe: pd.DataFrame,
    price_index: pd.DataFrame,
) -> pd.DataFrame:
    index_frame = price_index.copy()
    index_frame["quarter_start"] = index_frame["quarter"].apply(quarter_to_timestamp)
    index_frame = index_frame.sort_values("quarter_start").reset_index(drop=True)
    index_frame["year_quarter"] = index_frame["quarter"]

    reference_index = float(index_frame["index"].iloc[-1])

    merged = dataframe.copy()
    merged["year_quarter"] = merged["month"].apply(month_to_quarter)
    merged = merged.merge(
        index_frame[["year_quarter", "index"]],
        how="left",
        on="year_quarter",
    )
    merged = merged.rename(columns={"index": "price_index"})

    if merged["price_index"].isna().any():
        missing_rows = int(merged["price_index"].isna().sum())
        raise ValueError(f"Price index merge failed for {missing_rows} rows.")

    merged["inflation_adjustment_factor"] = reference_index / merged["price_index"]
    merged["corrected_resale_price"] = (
        merged["resale_price"] * merged["inflation_adjustment_factor"]
    )
    merged["corrected_price_per_sqm"] = (
        merged["corrected_resale_price"] / merged["floor_area_sqm"]
    )
    return merged


def add_group_rolling_features(
    dataframe: pd.DataFrame,
    group_column: str,
    target_column: str,
    windows: tuple[int, ...],
) -> tuple[pd.DataFrame, list[str]]:
    monthly_stats = (
        dataframe.groupby([group_column, "month"])[target_column]
        .agg(["mean", "std", "max", "min", "count"])
        .reset_index()
        .sort_values([group_column, "month"])
        .reset_index(drop=True)
    )
    monthly_stats["std"] = monthly_stats["std"].fillna(0.0)

    rolling_columns: list[str] = []
    grouped = monthly_stats.groupby(group_column, group_keys=False)
    reducers = {
        "mean": "mean",
        "std": "mean",
        "max": "max",
        "min": "min",
        "count": "sum",
    }

    for window in windows:
        for base_column, reducer in reducers.items():
            feature_name = (
                f"prior_{window}_{group_column}_{base_column}_corrected_price_per_sqm"
            )
            rolling_columns.append(feature_name)
            monthly_stats[feature_name] = grouped[base_column].transform(
                lambda series, reducer=reducer, window=window: getattr(
                    series.rolling(window=window, min_periods=1), reducer
                )().shift(1)
            )

    feature_frame = monthly_stats[[group_column, "month", *rolling_columns]]
    enriched = dataframe.merge(feature_frame, how="left", on=[group_column, "month"])
    return enriched, rolling_columns


def build_feature_dataset(
    transactions: pd.DataFrame,
    price_index: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, list[str]]:
    minimum_month = pd.Timestamp(config.minimum_transaction_month)
    dataframe = transactions.loc[transactions["month"] >= minimum_month].copy()
    dataframe = dataframe.sort_values("month").reset_index(drop=True)
    dataframe["month_num"] = dataframe["month"].dt.month.astype(int)
    dataframe["quarter"] = dataframe["month"].dt.quarter.astype(int)
    dataframe["transaction_year"] = dataframe["month"].dt.year.astype(int)
    dataframe["remaining_lease_years"] = dataframe["remaining_lease"].apply(parse_remaining_lease)
    dataframe["lease_age"] = (
        dataframe["transaction_year"] + ((dataframe["month_num"] - 1) / 12.0)
    ) - dataframe["lease_commence_date"]
    dataframe["storey_midpoint"] = dataframe["storey_range"].apply(parse_storey_midpoint)
    dataframe["flat_type_room_count"] = dataframe["flat_type"].apply(parse_flat_type_room_count)

    dataframe = attach_market_index(dataframe, price_index)
    dataframe["town_street"] = (
        dataframe["town"].fillna("Unknown") + " | " + dataframe["street_name"].fillna("Unknown")
    )
    dataframe["town_street_block"] = dataframe["town_street"] + " | " + dataframe["block"].fillna(
        "Unknown"
    )

    rolling_columns: list[str] = []
    for group_column in config.rolling_group_columns:
        dataframe, generated_columns = add_group_rolling_features(
            dataframe=dataframe,
            group_column=group_column,
            target_column="corrected_price_per_sqm",
            windows=config.rolling_windows,
        )
        rolling_columns.extend(generated_columns)

    for column in config.categorical_features:
        dataframe[column] = dataframe[column].fillna("Unknown").astype(str)

    feature_columns = list(config.categorical_features) + list(BASE_NUMERIC_FEATURES) + rolling_columns
    return dataframe, feature_columns
