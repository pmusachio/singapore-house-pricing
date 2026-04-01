# Project Summary

## Dataset split

| Split | Rows |
| --- | ---: |
| Train | 92,080 |
| Validation | 12,816 |
| Test | 12,110 |

## Key metrics

| Split | RMSE | MAE | MAPE (%) | Coverage | Mean interval width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 117,185 | 76,372 | 11.44 | 0.806 | 260,048 |
| Test | 130,584 | 89,579 | 13.05 | 0.743 | 261,070 |

## Notes

- Feature count used for training: 45
- Evaluation is time-based to reduce leakage from future transactions.
- Prediction intervals come from three independent quantile boosting models.
- The detailed metrics JSON and row-level predictions are available in this folder.