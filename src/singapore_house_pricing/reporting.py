from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_prediction_interval_sample(
    predictions: pd.DataFrame,
    split_name: str,
    output_path: Path,
) -> None:
    if predictions.empty:
        return

    sample = predictions.copy()
    if len(sample) > 250:
        sample = sample.sample(250, random_state=42)

    sample = sample.sort_values("actual_price").reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    x_axis = range(len(sample))
    plt.plot(x_axis, sample["actual_price"], label="Observed price", linewidth=2)
    plt.plot(x_axis, sample["median_prediction"], label="Predicted median", linewidth=2)
    plt.fill_between(
        x_axis,
        sample["lower_prediction"],
        sample["upper_prediction"],
        alpha=0.2,
        label="Prediction interval",
    )
    plt.title(f"{split_name.title()} sample: observed vs predicted interval")
    plt.xlabel("Sampled transactions sorted by observed price")
    plt.ylabel("Price (SGD)")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_project_summary(
    metrics_by_split: dict[str, dict[str, float]],
    split_sizes: dict[str, int],
    feature_count: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Project Summary",
        "",
        "## Dataset split",
        "",
        "| Split | Rows |",
        "| --- | ---: |",
    ]

    for split_name, size in split_sizes.items():
        lines.append(f"| {split_name.title()} | {size:,} |")

    lines.extend(
        [
            "",
            "## Key metrics",
            "",
            "| Split | RMSE | MAE | MAPE (%) | Coverage | Mean interval width |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for split_name, values in metrics_by_split.items():
        lines.append(
            "| "
            f"{split_name.title()} | "
            f"{values['rmse']:,.0f} | "
            f"{values['mae']:,.0f} | "
            f"{values['mape']:.2f} | "
            f"{values['coverage']:.3f} | "
            f"{values['mean_interval_width']:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Feature count used for training: {feature_count}",
            "- Evaluation is time-based to reduce leakage from future transactions.",
            "- Prediction intervals come from three independent quantile boosting models.",
            "- The detailed metrics JSON and row-level predictions are available in this folder.",
        ]
    )

    output_path.write_text("\n".join(lines))
