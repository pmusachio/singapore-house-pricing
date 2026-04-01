# Singapore House Pricing

Production-grade data science case study for probabilistic valuation of Singapore HDB resale flats.

This repository started as a notebook-heavy class project. It has now been refactored into a reproducible
portfolio project with a proper Python package, time-based evaluation, quantile prediction intervals,
tests, and documented outputs.

## Business goal

Estimate a fair resale price range for public housing units in Singapore using property characteristics,
local transaction history, and a macro housing price index. The model produces lower, median, and upper
price estimates instead of a single point forecast.

## What changed from the legacy version

- Replaced brittle notebook-only execution with a package under `src/`
- Removed broken path assumptions such as `week_2/...`
- Rebuilt feature engineering as reusable Python code
- Added a CLI for dataset preparation and model training
- Added a time-based split that reflects real deployment conditions
- Added tests for critical feature-engineering logic
- Documented outputs, metrics, and project structure for portfolio use

## Dataset coverage

- Raw data available in the repository: January 2017 to June 2024
- Modeling window used by the production pipeline: January 2020 to June 2024
- Default split:
  - Train: January 2020 to June 2023
  - Validation: July 2023 to December 2023
  - Test: January 2024 to June 2024

## Project structure

```text
.
├── data/
│   └── raw/
├── reports/
├── scripts/
├── src/singapore_house_pricing/
└── tests/
```

Generated directories such as `data/processed/`, `models/`, and `reports/` are created automatically
when you run the pipeline.

## Quickstart

The simplest way to run everything from setup to results is:

```bash
./scripts/run_project.sh
```

This command will create `.venv`, install dependencies, prepare the data, train the model on a smoke-test
sample, and print the summary at the end.

Useful variations:

```bash
./scripts/run_project.sh --full
./scripts/run_project.sh --with-tests
./scripts/run_project.sh --sample-rows 12000
```

If you prefer the manual flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
singapore-house-pricing prepare-data
singapore-house-pricing train --sample-rows 8000
```

If you prefer step-by-step execution:

```bash
singapore-house-pricing prepare-data
singapore-house-pricing train
pytest
```

For a fast smoke test that still exercises the training stack end to end:

```bash
singapore-house-pricing train --sample-rows 8000
```

## Outputs

Running the pipeline creates:

- `data/processed/sg_resale_flat_prices_engineered.csv`
- `data/processed/features_list.json`
- `models/quantile_10.joblib`
- `models/quantile_50.joblib`
- `models/quantile_90.joblib`
- `models/model_metadata.json`
- `reports/metrics.json`
- `reports/predictions.csv`
- `reports/project_summary.md`
- `reports/figures/*.png`

## Modeling approach

1. Load raw transactions and the quarterly housing price index.
2. Engineer leakage-safe features:
   - lease duration features
   - storey midpoint
   - market index and inflation adjustment factor
   - rolling historical price-per-square-meter signals by town and street
3. Train three quantile gradient boosting models for the 10th, 50th, and 90th percentiles.
4. Evaluate on out-of-time validation and test windows.
5. Save plots, metrics, predictions, and serialized models.

## Why this is portfolio-ready

- Clear business framing
- Reproducible codebase instead of ad-hoc notebook logic
- Time-aware model validation
- Uncertainty-aware predictions
- Test coverage for feature engineering
- Artifacts that support storytelling in interviews and portfolio pages

## Running notes

- The raw CSV files are already included in `data/raw/`, so you can run the project immediately.
- `./scripts/run_project.sh` is the recommended command for day-to-day use.
- Use `train --sample-rows 8000` for a fast smoke test.
- Use `train` without `--sample-rows` or `./scripts/run_project.sh --full` for the full dataset run.
