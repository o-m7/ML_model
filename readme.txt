# ML Trading System - Production-Ready Training Pipeline

A comprehensive, production-grade machine learning system for algorithmic trading with robust safeguards against common pitfalls like look-ahead bias, overfitting, and data snooping.

## Key Features

### ✅ Fixes from Original Code

1. **Proper Holdout Validation**: Thresholds selected on validation set, verified on test set (no data snooping)
2. **Embargo Periods**: Prevents information leakage in walk-forward validation
3. **Transaction Costs**: Spreads, commissions, and slippage included in all backtests
4. **Class Imbalance Handling**: Automatic class weighting for XGBoost
5. **Feature Correlation Filtering**: Removes redundant features
6. **Calibration Checking**: Ensures predicted probabilities are well-calibrated
7. **Data Quality Validation**: Comprehensive checks for gaps, outliers, and inconsistencies
8. **Realistic Trade Simulation**: Proper stop/TP mechanics with slippage
9. **Multi-timeframe Context**: No look-ahead bias in higher timeframe features
10. **Comprehensive Logging**: Full audit trail and reproducibility tracking

### 🚀 New Capabilities

- **Adaptive Stop/TP Levels**: ATR-based with regime adjustments
- **Position Correlation Management**: Prevents over-exposure to correlated pairs
- **Monte Carlo Ready**: Framework supports robustness testing
- **Model Versioning**: SHA hash tracking for data and model versions
- **Stability Analysis**: Detects overfitting in threshold selection
- **Feature Importance Tracking**: Automatic SHAP-ready importance extraction

## Installation

```bash
# Core dependencies
pip install pandas numpy polars xgboost scikit-learn

# Optional (highly recommended)
pip install ta-lib  # For faster technical indicators
pip install optuna  # For hyperparameter optimization

# Project structure
mkdir -p ~/Desktop/ML/{feature_store,models,reports,logs,cache}
```

## File Structure

```
.
├── config.py                  # Central configuration
├── data_utils.py             # Data loading & quality checks
├── features.py               # Feature engineering
├── labels.py                 # Label generation with cost modeling
├── walk_forward.py           # Time-series CV with embargo
├── backtest.py              # Realistic backtesting engine
├── threshold_selection.py    # Threshold optimization
├── model_training.py         # XGBoost training with imbalance handling
├── train.py                 # Main orchestration script
└── README.md
```

## Quick Start

### Single Symbol Training

```bash
# Train EURUSD 15-minute model
python train.py --symbol EURUSD --tf 15m

# Train XAUUSD 30-minute with custom risk
python train.py --symbol XAUUSD --tf 30m --risk-per-trade 0.005

# Train with session filter (London hours)
python train.py --symbol GBPUSD --tf 15m --session-start 8 --session-end 17
```

### Multi-Symbol Training

```bash
# Train all symbols on 60-minute timeframe
python train.py --symbol ALL --tf 60m --quick-scan

# Train specific symbols only
python train.py --symbol ALL --tf 15m --only EURUSD,GBPUSD,XAUUSD

# Skip certain symbols
python train.py --symbol ALL --tf 30m --skip NZDUSD,USDCAD
```

### Advanced Options

```bash
# Custom walk-forward periods
python train.py --symbol EURUSD --tf 15m \
  --train-months 9 \
  --test-months 2 \
  --validation-pct 0.25

# Regime and volatility filters
python train.py --symbol XAUUSD --tf 30m \
  --allow-regimes 1,2,3 \
  --vol-min 0.2 \
  --vol-max 0.8

# Disable multi-timeframe context (faster, less data needed)
python train.py --symbol EURUSD --tf 15m --no-mtf-context

# Date range filtering
python train.py --symbol GBPUSD --tf 15m \
  --start-date 2023-01-01 \
  --end-date 2024-12-31
```

## Configuration

Edit `config.py` to customize:

### Path Configuration
```python
paths:
  data_root: Path to parquet files
  model_dir: Where models are saved
  reports_dir: JSON reports output
```

### Trading Parameters
```python
trading:
  equity_start: 100000
  risk_per_trade_default: 0.01  # 1% per trade
  max_dd_pct_limit: 0.04        # 4% drawdown cap
  spread_pips: {...}            # Symbol-specific spreads
  commission_per_lot: 7.0       # USD per round-turn
```

### Model Parameters
```python
model:
  train_months: 6.0
  test_months: 1.0
  validation_pct: 0.20  # 20% of OOS for threshold selection
  handle_imbalance: True
  enable_hp_tuning: False  # Set True for Optuna
```

## Data Requirements

### Parquet File Format

Expected structure:
```
~/Desktop/ML/feature_store/
└── EURUSD/
    ├── EURUSD_1T.parquet   # 1-minute
    ├── EURUSD_5T.parquet   # 5-minute
    └── EURUSD_15T.parquet  # 15-minute
```

Required columns:
- `timestamp` (or `time`, `datetime`, etc.)
- `open`, `high`, `low`, `close`
- `volume` (optional, but recommended)

### Minimum Data Requirements

| Timeframe | Minimum Bars | Recommended |
|-----------|-------------|-------------|
| 1m        | 50,000      | 100,000+    |
| 5m        | 20,000      | 50,000+     |
| 15m       | 10,000      | 30,000+     |
| 30m       | 8,000       | 20,000+     |
| 60m       | 6,000       | 15,000+     |
| 240m      | 4,000       | 10,000+     |

## Understanding the Training Process

### Step 1: Data Loading & Validation
- Loads parquet file
- Detects and converts timestamp formats
- Removes duplicates and outliers
- Validates OHLC consistency
- Checks for time gaps

**Quality Gates:**
- ✅ Max 5% missing data allowed
- ✅ Minimum 10,000 rows required
- ✅ No price inconsistencies (high < low, etc.)

### Step 2: Feature Engineering
- Calculates 50+ technical indicators
- Adds multi-timeframe context (optional)
- Creates feature interactions
- Filters highly correlated features

**Safeguards:**
- ✅ Forward-fill only for higher timeframe context
- ✅ No look-ahead bias in calculations
- ✅ Removes features with >95% correlation

### Step 3: Label Generation
- Calculates adaptive stop/TP levels based on ATR
- Includes transaction costs in labels
- Adjusts for market regime
- Vectorized for speed

**Realistic Modeling:**
- ✅ Spread costs deducted from P&L
- ✅ Commission included
- ✅ Slippage modeled
- ✅ Regime-adaptive stop sizes

### Step 4: Feature Selection
- Removes features with <98% coverage
- Filters by correlation
- Ranks by mutual information (if target provided)
- Limits to top 100 features

### Step 5: Walk-Forward Splitting
- Creates time-based train/val/test splits
- Applies embargo periods between sets
- Validates no temporal leakage
- Minimum 3 folds required

**Structure:**
```
[--- Train ---][Embargo][-- Val --][Embargo][-- Test --]
     6 months      X      0.4 mo      X       0.6 mo
```

### Step 6: Model Training
- Trains XGBoost classifiers (long/short)
- Trains XGBoost regressor (expected return)
- Handles class imbalance automatically
- Uses early stopping on validation set
- Checks probability calibration

**Per Fold:**
- Long classifier (AUC, log loss)
- Short classifier (AUC, log loss)
- ExpR regressor (RMSE)
- Feature importance extraction

### Step 7: Threshold Selection (CRITICAL)
- Uses **validation set only** for threshold search
- Tests grid of probability/expR combinations
- Scores candidates by Sharpe, PF, trade count
- Validates winner on **test set**
- Detects overfitting (validation vs test degradation)

**Scoring Function:**
```python
score = trade_count_penalty * (
    0.4 * sharpe +
    0.3 * (profit_factor - 1) +
    0.3 * avg_R
)
```

### Step 8: Final Model Training
- Trains on ALL available data
- Uses selected thresholds from step 7
- Saves with version hash and metadata

### Step 9: Reporting
- Saves model with full metadata
- Generates threshold report
- Exports feature importance
- Creates training summary JSON

## Output Files

### Model File
```
~/Desktop/ML/models/EURUSD_15m_xgb_20241028_142530_a3f5b8c9.pkl
```

Contains:
- Trained XGBoost models (long, short, expR)
- Feature column names
- Feature importance scores
- Training metrics
- Calibration data
- Full metadata for reproducibility

### Reports
```
~/Desktop/ML/reports/EURUSD/15m/
├── thresholds_20241028_142530.json  # Selected thresholds + metrics
├── feature_importance_20241028_142530.json
└── summary_20241028_142530.json     # Quick reference
```

## Loading and Using Models

```python
from model_training import ModelBundle
import pandas as pd

# Load model
bundle = ModelBundle.load("path/to/model.pkl")

# Prepare features (must match training features exactly)
X = df[bundle.feature_columns]

# Generate predictions
prob_long, prob_short, expR = bundle.predict(X)

# Apply thresholds from metadata
thr_long = bundle.metadata['thresholds']['threshold_long']
thr_short = bundle.metadata['thresholds']['threshold_short']
expr_min = bundle.metadata['thresholds']['expr_min']

# Filter signals
long_signals = (prob_long >= thr_long) & (expR >= expr_min)
short_signals = (prob_short >= thr_short) & (expR >= expr_min)
```

## Performance Metrics Explained

### Validation Metrics
- **AUC**: Classifier discrimination ability (>0.55 is tradeable)
- **ECE**: Calibration error (<0.10 is well-calibrated)
- **Win Rate**: Percentage of winning trades (>45% target)
- **Profit Factor**: Gross profit / gross loss (>1.3 target)
- **Sharpe Ratio**: Risk-adjusted return per trade (>0.5 target)
- **Max Drawdown**: Peak-to-trough equity decline (<4% cap)

### Test Metrics (True OOS)
Same as validation but on unseen test set. 

**Warning Signs:**
- Test Sharpe < 70% of validation Sharpe → Overfitting
- Test PF < validation PF by >20% → Unstable
- Test trade count << validation → Selection bias

## Troubleshooting

### No Viable Thresholds Found
```
Possible causes:
1. Model predictions not calibrated (all close to 0.5)
2. Transaction costs too high relative to signal strength
3. Filters too restrictive (vol, regime, session)
4. Insufficient data for timeframe

Solutions:
- Check AUC scores (should be >0.52)
- Reduce cooldown_bars
- Widen vol_min/vol_max range
- Use --quick-scan for faster iteration
- Try lower timeframe with more data
```

### Insufficient Data Error
```
Solutions:
- Reduce --train-months and --test-months
- Use --no-mtf-context to reduce warmup period
- Check data quality (gaps, missing files)
```

### High Validation-Test Degradation
```
Warning: Test Sharpe significantly worse than validation

This indicates overfitting. Try:
- Simpler feature set (--max-features 50)
- Stronger regularization in config.py
- Longer embargo periods
- More folds (reduce train/test months)
```

### Memory Issues
```
For large datasets (>500k bars):

1. Disable MTF context: --no-mtf-context
2. Reduce max features: --max-features 50
3. Use quick scan: --quick-scan
4. Process symbols sequentially (don't use --symbol ALL)
```

## Best Practices

### ✅ DO
1. Always use holdout validation (already built-in)
2. Check test set performance before deploying
3. Monitor calibration (ECE < 0.10)
4. Include transaction costs in all backtests
5. Use embargo periods (automatic)
6. Version control your data and models
7. Paper trade before going live
8. Monitor model degradation in production

### ❌ DON'T
1. Select thresholds on test set (script prevents this)
2. Ignore data quality warnings
3. Use models with AUC < 0.52
4. Deploy models with high validation-test degradation
5. Skip regime/volatility filters if they improve metrics
6. Over-optimize on one time period
7. Ignore correlation between strategies

## Advanced Topics

### Adding Custom Features

Edit `features.py`:
```python
def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Your custom calculations here
    df["my_indicator"] = ...
    
    return df
```

Then call in `build_features()` method.

### Custom Risk Models

Edit `config.py` and `labels.py`:
```python
# In config.py
trading:
  stop_atr_mult_min: 0.8
  stop_atr_mult_max: 1.5
  
# In labels.py, modify generate_adaptive_stops_targets()
```

### Monte Carlo Robustness Testing

```python
from backtest import backtest_signals
import numpy as np

# Resample trades with replacement
for i in range(1000):
    resampled_trades = np.random.choice(all_trades, len(all_trades), replace=True)
    # Calculate metrics on resampled trades
    # Build distribution of outcomes
```

## FAQ

**Q: How long does training take?**
A: 15m timeframe with 30k bars takes ~5-10 minutes on modern hardware. Higher timeframes or more data take longer.

**Q: Can I use this for crypto/stocks?**
A: Yes, but update `spread_pips` and `commission_per_lot` in config.py for your asset class.

**Q: What's the minimum win rate needed?**
A: Depends on profit factor. With 1.5:1 RR, you need ~45% WR. With 2:1 RR, 40% works.

**Q: Should I use multi-timeframe context?**
A: Yes for <60m timeframes. Disable for 240m (no higher TF available).

**Q: How often should I retrain?**
A: Monitor test metrics monthly. Retrain if Sharpe drops >30% or new regime emerges.

**Q: Can I modify the threshold after training?**
A: You can, but it defeats the validation process. Better to retrain with different filters.

## Citation & License

If you use this code in research or production:
```
ML Trading System - Production-Grade Training Pipeline
Built with proper validation, embargo periods, and transaction cost modeling
2024
```

## Support

For issues, questions, or contributions:
1. Check this README first
2. Review config.py comments
3. Enable verbose logging in train.py
4. Check generated JSON reports in ~/Desktop/ML/reports/

---

**Remember:** Past performance ≠ future results. Always paper trade first!