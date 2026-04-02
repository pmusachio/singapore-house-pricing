from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    project_dir: Path = PROJECT_DIR
    raw_transactions: Path = (
        PROJECT_DIR / "data" / "raw" / "sg-resale-flat-prices-2017-onwards.csv"
    )
    raw_price_index: Path = PROJECT_DIR / "data" / "raw" / "sg_price_index.csv"
    processed_dataset: Path = (
        PROJECT_DIR / "data" / "processed" / "sg_resale_flat_prices_engineered.csv"
    )
    features_path: Path = PROJECT_DIR / "data" / "processed" / "features_list.json"
    models_dir: Path = PROJECT_DIR / "models"
    reports_dir: Path = PROJECT_DIR / "reports"
    figures_dir: Path = PROJECT_DIR / "reports" / "figures"
    metrics_path: Path = PROJECT_DIR / "reports" / "metrics.json"
    predictions_path: Path = PROJECT_DIR / "reports" / "predictions.csv"
    summary_path: Path = PROJECT_DIR / "reports" / "project_summary.md"
    model_metadata_path: Path = PROJECT_DIR / "models" / "model_metadata.json"

    def ensure_directories(self) -> None:
        for path in (
            self.processed_dataset.parent,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SplitConfig:
    train_end: str = "2023-07-01"
    validation_end: str = "2024-01-01"


@dataclass(frozen=True)
class FeatureConfig:
    minimum_transaction_month: str = "2020-01-01"
    rolling_windows: tuple[int, ...] = (1, 3, 6)
    rolling_group_columns: tuple[str, ...] = ("town", "town_street")
    categorical_features: tuple[str, ...] = ("town", "flat_type", "storey_range", "flat_model")


@dataclass(frozen=True)
class ModelConfig:
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_estimators: int = 40
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 20
    subsample: float = 0.8
    random_state: int = 42


@dataclass(frozen=True)
class ProjectConfig:
    paths: Paths = field(default_factory=Paths)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_serialisable_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self), project_dir=self.paths.project_dir)


def _serialise(value: Any, project_dir: Path | None = None) -> Any:
    if isinstance(value, Path):
        if project_dir is not None:
            try:
                return str(value.relative_to(project_dir))
            except ValueError:
                pass
        return str(value)
    if isinstance(value, dict):
        return {key: _serialise(item, project_dir=project_dir) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item, project_dir=project_dir) for item in value]
    return value
