# Singapore House Pricing

Production-grade data science case study for probabilistic valuation of Singapore HDB resale flats.

## Business goal

Estimate a fair resale price range for public housing units in Singapore using property characteristics,
local transaction history, and a macro housing price index. The model produces lower, median, and upper
price estimates instead of a single point forecast.

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
./scripts/run_project.sh --full
```

This command will create `.venv`, install dependencies, prepare the data, train the model on a smoke-test
sample, and print the summary at the end.

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
