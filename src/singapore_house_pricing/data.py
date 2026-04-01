from __future__ import annotations

import json

import pandas as pd

from .config import Paths


STRING_COLUMNS = (
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "flat_model",
    "remaining_lease",
)


def load_raw_transactions(paths: Paths) -> pd.DataFrame:
    dataframe = pd.read_csv(paths.raw_transactions)
    dataframe["month"] = pd.to_datetime(dataframe["month"], format="%Y-%m", errors="coerce")
    dataframe["lease_commence_date"] = pd.to_numeric(
        dataframe["lease_commence_date"], errors="coerce"
    )
    dataframe["floor_area_sqm"] = pd.to_numeric(dataframe["floor_area_sqm"], errors="coerce")
    dataframe["resale_price"] = pd.to_numeric(dataframe["resale_price"], errors="coerce")

    for column in STRING_COLUMNS:
        dataframe[column] = dataframe[column].astype("string").str.strip()

    dataframe = dataframe.dropna(subset=["month", "floor_area_sqm", "resale_price"])
    return dataframe.sort_values("month").reset_index(drop=True)


def load_price_index(paths: Paths) -> pd.DataFrame:
    dataframe = pd.read_csv(paths.raw_price_index)
    dataframe["quarter"] = dataframe["quarter"].astype("string").str.strip()
    dataframe["index"] = pd.to_numeric(dataframe["index"], errors="coerce")
    dataframe = dataframe.dropna(subset=["quarter", "index"])
    return dataframe.reset_index(drop=True)


def save_processed_dataset(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    paths: Paths,
) -> None:
    paths.ensure_directories()
    dataframe.to_csv(paths.processed_dataset, index=False)
    paths.features_path.write_text(json.dumps(feature_columns, indent=2))


def load_processed_dataset(paths: Paths) -> tuple[pd.DataFrame, list[str]]:
    dataframe = pd.read_csv(paths.processed_dataset, parse_dates=["month"])
    feature_columns = json.loads(paths.features_path.read_text())
    return dataframe, feature_columns
