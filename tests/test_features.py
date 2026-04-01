import pandas as pd
import pytest

from singapore_house_pricing.features import (
    add_group_rolling_features,
    month_to_quarter,
    parse_flat_type_room_count,
    parse_remaining_lease,
    parse_storey_midpoint,
    quarter_to_timestamp,
)


def test_parse_remaining_lease_handles_common_formats() -> None:
    assert parse_remaining_lease("61 years 04 months") == pytest.approx(61 + (4 / 12))
    assert parse_remaining_lease("61 years") == pytest.approx(61.0)
    assert parse_remaining_lease("8 months") == pytest.approx(8 / 12)


def test_parse_storey_midpoint_returns_average_floor() -> None:
    assert parse_storey_midpoint("10 TO 12") == pytest.approx(11.0)
    assert parse_storey_midpoint("01 TO 03") == pytest.approx(2.0)


def test_parse_flat_type_room_count_extracts_leading_number() -> None:
    assert parse_flat_type_room_count("4 ROOM") == pytest.approx(4.0)
    assert pd.isna(parse_flat_type_room_count("EXECUTIVE"))


def test_quarter_conversions_are_consistent() -> None:
    january = quarter_to_timestamp("2024-Q1")
    assert january == pd.Timestamp("2024-01-01")
    assert month_to_quarter(pd.Timestamp("2024-06-01")) == "2024-Q2"


def test_rolling_features_use_only_past_observations() -> None:
    dataframe = pd.DataFrame(
        {
            "town": ["A", "A", "A", "B"],
            "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-01"]),
            "corrected_price_per_sqm": [100.0, 200.0, 300.0, 500.0],
        }
    )

    enriched, columns = add_group_rolling_features(
        dataframe=dataframe,
        group_column="town",
        target_column="corrected_price_per_sqm",
        windows=(1, 2),
    )

    march_row = enriched.loc[
        (enriched["town"] == "A") & (enriched["month"] == pd.Timestamp("2024-03-01"))
    ].iloc[0]
    january_row = enriched.loc[
        (enriched["town"] == "A") & (enriched["month"] == pd.Timestamp("2024-01-01"))
    ].iloc[0]

    assert "prior_1_town_mean_corrected_price_per_sqm" in columns
    assert pd.isna(january_row["prior_1_town_mean_corrected_price_per_sqm"])
    assert march_row["prior_1_town_mean_corrected_price_per_sqm"] == pytest.approx(200.0)
    assert march_row["prior_2_town_mean_corrected_price_per_sqm"] == pytest.approx(150.0)
