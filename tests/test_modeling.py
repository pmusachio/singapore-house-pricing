import numpy as np
import pandas as pd

from singapore_house_pricing.config import SplitConfig
from singapore_house_pricing.modeling import enforce_interval_order, split_dataset_by_time


def test_split_dataset_by_time_respects_boundaries() -> None:
    dataframe = pd.DataFrame(
        {
            "month": pd.to_datetime(["2023-06-01", "2023-07-01", "2024-01-01"]),
            "resale_price": [100_000, 200_000, 300_000],
        }
    )

    split_frames = split_dataset_by_time(
        dataframe,
        SplitConfig(train_end="2023-07-01", validation_end="2024-01-01"),
    )

    assert len(split_frames["train"]) == 1
    assert len(split_frames["validation"]) == 1
    assert len(split_frames["test"]) == 1


def test_enforce_interval_order_sorts_crossed_predictions() -> None:
    lower, median, upper = enforce_interval_order(
        lower=np.array([300.0]),
        median=np.array([100.0]),
        upper=np.array([200.0]),
    )

    assert lower[0] == 100.0
    assert median[0] == 200.0
    assert upper[0] == 300.0
