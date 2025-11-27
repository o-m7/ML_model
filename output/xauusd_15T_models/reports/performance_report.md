# XAUUSD Model Training - Performance Report

Generated: 2025-11-17 15:47:11.089055

## Executive Summary

| model    |   deployment_ready |   profit_factor |   win_rate |   max_drawdown_pct |   sharpe |   r_multiple |   total_trades |
|:---------|-------------------:|----------------:|-----------:|-------------------:|---------:|-------------:|---------------:|
| catboost |                  0 |          3.3835 |     0.7358 |             1.8699 |   0.5188 |       1.1582 |        215.841 |
| lightgbm |                  0 |          4.622  |     0.788  |             1.2976 |   0.6579 |       1.1711 |        213.849 |
| xgboost  |                  0 |          7.924  |     0.8572 |             0.7211 |   0.9058 |       1.2175 |        212.524 |


## Walk-Forward Validation Results

|                       |   ('profit_factor', 'mean') |   ('profit_factor', 'std') |   ('profit_factor', 'min') |   ('profit_factor', 'max') |   ('win_rate', 'mean') |   ('win_rate', 'std') |   ('max_drawdown_pct', 'mean') |   ('max_drawdown_pct', 'max') |   ('sharpe', 'mean') |   ('sharpe', 'std') |   ('total_trades', 'sum') |
|:----------------------|----------------------------:|---------------------------:|---------------------------:|---------------------------:|-----------------------:|----------------------:|-------------------------------:|------------------------------:|---------------------:|--------------------:|--------------------------:|
| ('catboost', 'test')  |                      3.3835 |                     1.0909 |                     1.5696 |                     7.6139 |                 0.7358 |                0.0498 |                         1.8699 |                        5.1217 |               0.5188 |              0.1276 |                     27196 |
| ('catboost', 'train') |                      3.3136 |                     0.5695 |                     2.2401 |                     5.0136 |                 0.7394 |                0.0216 |                         1.6135 |                        3.4219 |               0.5069 |              0.064  |                    108496 |
| ('catboost', 'val')   |                      3.4013 |                     1.1055 |                     1.5696 |                     7.6139 |                 0.7361 |                0.0501 |                         1.8458 |                        5.1217 |               0.5214 |              0.131  |                     27153 |
| ('lightgbm', 'test')  |                      4.622  |                     1.6126 |                     1.9765 |                     9.9444 |                 0.788  |                0.0443 |                         1.2976 |                        3.8939 |               0.6579 |              0.1402 |                     26945 |
| ('lightgbm', 'train') |                      4.4871 |                     0.8231 |                     3.0315 |                     6.779  |                 0.7914 |                0.0212 |                         1.1274 |                        2.1981 |               0.6359 |              0.0787 |                    107411 |
| ('lightgbm', 'val')   |                      4.6517 |                     1.6163 |                     1.9765 |                     9.9444 |                 0.7887 |                0.0443 |                         1.274  |                        3.8939 |               0.6617 |              0.1422 |                     26888 |
| ('xgboost', 'test')   |                      7.924  |                     2.966  |                     2.8102 |                    18.7088 |                 0.8572 |                0.0383 |                         0.7211 |                        3.7972 |               0.9058 |              0.1711 |                     26778 |
| ('xgboost', 'train')  |                      7.6693 |                     1.5095 |                     5.0592 |                    12.1725 |                 0.8605 |                0.0179 |                         0.5528 |                        1.2134 |               0.8626 |              0.1005 |                    106864 |
| ('xgboost', 'val')    |                      7.9696 |                     2.9402 |                     2.8102 |                    18.7088 |                 0.858  |                0.0376 |                         0.6954 |                        3.7972 |               0.9105 |              0.1697 |                     26751 |


## Model Comparison

| model    |   profit_factor |   win_rate |   max_drawdown_pct |   sharpe |   r_multiple |   expected_value_dollar |   total_trades |
|:---------|----------------:|-----------:|-------------------:|---------:|-------------:|------------------------:|---------------:|
| xgboost  |          7.924  |     0.8572 |             0.7211 |   0.9058 |       1.2175 |                  0.0016 |          26778 |
| lightgbm |          4.622  |     0.788  |             1.2976 |   0.6579 |       1.1711 |                  0.0013 |          26945 |
| catboost |          3.3835 |     0.7358 |             1.8699 |   0.5188 |       1.1582 |                  0.0011 |          27196 |


## Deployment Recommendations

**WARNING**: No models currently meet all deployment criteria.

Review training data, features, or labeling strategy.


## Next Steps

1. Review feature importance plots

2. Examine equity curves for stability

3. Check prediction calibration

4. Test on live paper trading

5. Monitor for data drift
