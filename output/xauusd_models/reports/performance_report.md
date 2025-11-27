# XAUUSD Model Training - Performance Report

Generated: 2025-11-17 11:39:37.963941

## Executive Summary

| model    |   deployment_ready |   profit_factor |   win_rate |   max_drawdown_pct |   sharpe |   r_multiple |   total_trades |
|:---------|-------------------:|----------------:|-----------:|-------------------:|---------:|-------------:|---------------:|
| catboost |                  0 |          1.1995 |     0.4914 |            23.6085 |   0.0649 |       1.2029 |        513     |
| lightgbm |                  0 |          1.1995 |     0.4914 |            23.6085 |   0.0649 |       1.2029 |        513     |
| xgboost  |                  0 |          4.1123 |     0.7363 |             4.2466 |   0.6012 |       1.2384 |        280.333 |


## Walk-Forward Validation Results

|                       |   ('profit_factor', 'mean') |   ('profit_factor', 'std') |   ('profit_factor', 'min') |   ('profit_factor', 'max') |   ('win_rate', 'mean') |   ('win_rate', 'std') |   ('max_drawdown_pct', 'mean') |   ('max_drawdown_pct', 'max') |   ('sharpe', 'mean') |   ('sharpe', 'std') |   ('total_trades', 'sum') |
|:----------------------|----------------------------:|---------------------------:|---------------------------:|---------------------------:|-----------------------:|----------------------:|-------------------------------:|------------------------------:|---------------------:|--------------------:|--------------------------:|
| ('catboost', 'test')  |                      1.1995 |                     0.3583 |                     0.5931 |                     2.1237 |                 0.4914 |                0.0679 |                        23.6085 |                       53.9507 |               0.0649 |              0.133  |                     13851 |
| ('catboost', 'train') |                      1.1276 |                     0.0762 |                     0.9606 |                     1.27   |                 0.4803 |                0.0319 |                        32.1738 |                       89.0109 |               0.0519 |              0.0305 |                     82760 |
| ('catboost', 'val')   |                      1.1685 |                     0.353  |                     0.5931 |                     2.1237 |                 0.4827 |                0.0674 |                        23.9652 |                       53.9507 |               0.0531 |              0.1317 |                     13827 |
| ('lightgbm', 'test')  |                      1.1995 |                     0.3583 |                     0.5931 |                     2.1237 |                 0.4914 |                0.0679 |                        23.6085 |                       53.9507 |               0.0649 |              0.133  |                     13851 |
| ('lightgbm', 'train') |                      1.1276 |                     0.0762 |                     0.9606 |                     1.27   |                 0.4803 |                0.0319 |                        32.1738 |                       89.0109 |               0.0519 |              0.0305 |                     82760 |
| ('lightgbm', 'val')   |                      1.1685 |                     0.353  |                     0.5931 |                     2.1237 |                 0.4827 |                0.0674 |                        23.9652 |                       53.9507 |               0.0531 |              0.1317 |                     13827 |
| ('xgboost', 'test')   |                      4.1123 |                     2.16   |                     1.3135 |                     8.3136 |                 0.7363 |                0.0847 |                         4.2466 |                        9.8653 |               0.6012 |              0.2588 |                      7569 |
| ('xgboost', 'train')  |                      3.5993 |                     0.7213 |                     2.3541 |                     4.7883 |                 0.7342 |                0.0319 |                         2.1537 |                        2.9561 |               0.5632 |              0.0819 |                     49050 |
| ('xgboost', 'val')    |                      4.1764 |                     2.0985 |                     1.3808 |                     8.3136 |                 0.7407 |                0.0781 |                         4.004  |                        9.6463 |               0.615  |              0.2422 |                      7783 |


## Model Comparison

| model    |   profit_factor |   win_rate |   max_drawdown_pct |   sharpe |   r_multiple |   expected_value_dollar |   total_trades |
|:---------|----------------:|-----------:|-------------------:|---------:|-------------:|------------------------:|---------------:|
| xgboost  |          4.1123 |     0.7363 |             4.2466 |   0.6012 |       1.2384 |                  0.0024 |           7569 |
| catboost |          1.1995 |     0.4914 |            23.6085 |   0.0649 |       1.2029 |                  0.0003 |          13851 |
| lightgbm |          1.1995 |     0.4914 |            23.6085 |   0.0649 |       1.2029 |                  0.0003 |          13851 |


## Deployment Recommendations

**WARNING**: No models currently meet all deployment criteria.

Review training data, features, or labeling strategy.


## Next Steps

1. Review feature importance plots

2. Examine equity curves for stability

3. Check prediction calibration

4. Test on live paper trading

5. Monitor for data drift
