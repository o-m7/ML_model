# Citadel Metals Training System
**Production-Grade ML Pipeline for XAUUSD/XAGUSD Algorithmic Trading**

---

## Overview

The Citadel training system is a **complete, prop-firm-grade ML pipeline** designed to train robust algorithmic trading models for precious metals (XAUUSD and XAGUSD).

### Key Features

✅ **Three Strategy Archetypes**:
- Short-term mean reversion (5T, 15T)
- Volatility breakouts (5T, 15T, 30T)
- Trend following with regime filters (30T, 1H)

✅ **Zero Data Leakage**:
- Proper temporal train/val/test splits
- Forward-safe feature engineering
- No future information in any feature

✅ **Realistic Trading Simulation**:
- TP/SL based on ATR multiples
- Spread + commission costs per symbol
- R-multiple and expectancy calculations

✅ **Strict Validation**:
- Only saves models passing all thresholds
- Minimum 200 trades on test set
- Win rate ≥ 48%, Profit factor ≥ 1.3
- Max drawdown ≤ 20%, Expectancy ≥ 0.1

---

## Quick Start

### 1. Ensure Data Exists

The system expects data in:
```
feature_store/
├── XAUUSD/
│   └── XAUUSD_5T.parquet  (or any base timeframe)
└── XAGUSD/
    └── XAGUSD_5T.parquet
```

Data format: DatetimeIndex with columns `open, high, low, close, volume`

### 2. Run Training

```bash
python citadel_training_system.py
```

### 3. Check Results

Models saved to:
```
models_citadel/
├── XAUUSD/
│   ├── XAUUSD_5T.pkl
│   ├── XAUUSD_15T.pkl
│   ├── XAUUSD_30T.pkl
│   └── XAUUSD_1H.pkl
└── XAGUSD/
    ├── XAGUSD_5T.pkl
    ├── XAGUSD_15T.pkl
    ├── XAGUSD_30T.pkl
    └── XAGUSD_1H.pkl
```

Plus summary: `models_citadel/training_summary.csv`

---

## Strategy Archetypes

### 1. Short-Term Mean Reversion

**Timeframes**: 5T, 15T

**Core Concept**: Price tends to revert to VWAP, EMA, or Bollinger mid after short-term overextensions.

**Key Features**:
```python
# Distance from VWAP in ATR multiples
dist_from_vwap = (close - vwap) / atr14

# Z-score vs 20-bar rolling mean
zscore_20 = (close - sma20) / std20

# Bollinger Band position (0 = lower band, 1 = upper band)
bb_position = (close - bb_lower) / (bb_upper - bb_lower)

# RSI extremes
rsi14  # <30 = oversold, >70 = overbought

# Stochastic
stoch_k, stoch_d
```

**Trading Logic**:
- Buy when price far below VWAP/BB mid + RSI oversold
- Sell when price snaps back to mean
- TP: 2.5x ATR, SL: 1.5x ATR

### 2. Volatility Breakout

**Timeframes**: 5T, 15T, 30T

**Core Concept**: Capture explosive moves when volatility expands sharply, especially at session opens.

**Key Features**:
```python
# Range expansion
range_vs_median = current_range / median_range_20

# ATR spike
atr_expansion = atr14 / atr50

# Breakout flags
breakout_high_10 = (close > max(high[-10]))
breakout_low_10 = (close < min(low[-10]))

# Session timing
london_open_hour  # 7-8 UTC
ny_open_hour      # 13-14 UTC
```

**Trading Logic**:
- Enter on range breakout + ATR spike
- Strong directional move during high-volume sessions
- Follow momentum until TP/SL hit

### 3. Trend Following

**Timeframes**: 30T, 1H

**Core Concept**: Enter with dominant trend after pullbacks, avoid choppy markets.

**Key Features**:
```python
# Multi-EMA trend
ema10, ema20, ema50, ema100, ema200
ema20_slope, ema50_slope

# Price vs EMAs
price_vs_ema20 = (close - ema20) / ema20

# Trend strength (like ADX)
trend_strength = abs(ema12 - ema26) / atr14

# MACD
macd, macd_signal, macd_hist

# Higher timeframe context (1H trend for 5T/15T)
htf_trend  # +1 = uptrend, -1 = downtrend
```

**Trading Logic**:
- Only trade in direction of higher timeframe trend
- Enter after pullback to EMA20/50 in strong trend
- Avoid trading when trend_strength is low (choppy)

---

## Feature Engineering

### 60+ Features Covering All Strategies

**Price & Returns** (5 features):
- `returns`, `log_returns`
- `ret_1`, `ret_3`, `ret_5`, `ret_10`, `ret_20`

**Volatility** (6 features):
- `vol_10`, `vol_20`, `vol_50`
- `atr14`, `atr20`, `atr50`

**Candle Patterns** (5 features):
- `range`, `body_pct`, `upper_wick_pct`, `lower_wick_pct`, `gap`

**Trend/Momentum** (15+ features):
- `ema10`, `ema20`, `ema50`, `ema100`, `ema200`
- `ema10_slope`, `ema20_slope`, etc.
- `price_vs_ema10`, `price_vs_ema20`, etc.
- `macd`, `macd_signal`, `macd_hist`
- `trend_strength`

**Mean Reversion** (8 features):
- `zscore_20`, `bb_mid`, `bb_upper`, `bb_lower`
- `bb_position`, `bb_width`
- `rsi14`, `stoch_k`, `stoch_d`
- `vwap_proxy`, `dist_from_vwap`

**Volatility/Breakout** (6 features):
- `range_vs_median`, `atr_expansion`
- `breakout_high_10`, `breakout_low_10`
- `dist_to_high_50`, `dist_to_low_50`

**Time-Based** (7 features):
- `hour`, `minute`, `day_of_week`, `minute_of_day`
- `london_session`, `ny_session`
- `london_open_hour`, `ny_open_hour`

**Volume** (3 features):
- `volume_ratio`, `volume_zscore`

**Higher Timeframe** (1 feature):
- `htf_trend` (hourly trend for intraday)

---

## Label Generation

### Classification Target

**3-class classification**:
- `0 = FLAT` (small move within ±threshold)
- `1 = LONG` (strong up move ≥ +threshold)
- `2 = SHORT` (strong down move ≤ -threshold)

**Threshold Calculation**:
```python
# Adaptive threshold scaled by recent volatility
threshold_pct = (0.8 * atr14) / close

# Label assignments
if future_return >= +threshold_pct:
    label = 1  # LONG

elif future_return <= -threshold_pct:
    label = 2  # SHORT

else:
    label = 0  # FLAT
```

**Forward Horizons** (prevents data leakage):
- 5T: 5 bars (25 minutes)
- 15T: 3 bars (45 minutes)
- 30T: 3 bars (90 minutes)
- 1H: 2 bars (2 hours)

**Typical Label Distribution**:
```
FLAT: 40-50%
LONG: 25-30%
SHORT: 25-30%
```

---

## Model Architecture

### XGBoost Classifier

**Hyperparameters** (optimized for trading):
```python
{
    'objective': 'multi:softprob',  # 3-class probabilities
    'num_class': 3,
    'max_depth': 6,              # Prevent overfitting
    'learning_rate': 0.05,       # Slow learning
    'n_estimators': 300,         # With early stopping
    'subsample': 0.8,            # Row sampling
    'colsample_bytree': 0.8,     # Feature sampling
    'min_child_weight': 5,       # Min samples per leaf
    'gamma': 0.1,                # Regularization
    'reg_alpha': 1.0,            # L1 regularization
    'reg_lambda': 2.0,           # L2 regularization
}
```

**Early Stopping**:
- 30 rounds on validation AUC
- Prevents overfitting to training data

---

## Data Splitting

### Temporal Train/Val/Test

**NO SHUFFLING** - Strict chronological order:

```
Full data: Jan 2023 ─────────────────────────────────> Dec 2024

Train (70%): Jan 2023 ════════════════> Jul 2024
Val (15%):                              Jul 2024 ══> Oct 2024
Test (15%):                                          Oct 2024 ─> Dec 2024
```

**Why this matters**:
- Model trained on past data only
- Validated on intermediate period
- Tested on most recent unseen data
- Mimics real-world: predict future from past

---

## Trade Simulation

### Realistic Cost Model

**Per Symbol Costs**:

**XAUUSD**:
- Spread: 20 points ($2.00 per lot)
- Commission: $7 per round turn
- Point value: $0.10 per point per lot

**XAGUSD**:
- Spread: 2 points ($0.02 per lot)
- Commission: $7 per round turn
- Point value: $0.01 per point per lot

### TP/SL Strategy

**ATR-Based Stops**:
```python
# Stop Loss: 1.5x ATR
sl_distance = 1.5 * atr14

# Take Profit: 2.5x ATR
tp_distance = 2.5 * atr14

# Risk/Reward = 2.5/1.5 = 1.67:1
```

### Trade Execution

**Entry**:
1. Model predicts LONG (1) or SHORT (2)
2. Enter at current close price
3. Set TP = entry ± 2.5 ATR
4. Set SL = entry ∓ 1.5 ATR

**Exit**:
1. Price hits TP → exit at TP (win)
2. Price hits SL → exit at SL (loss)
3. Signal changes → exit at current price (signal exit)
4. End of test period → close at last price (EOD)

**PnL Calculation**:
```python
# Gross PnL
if direction == 'long':
    gross_pnl = (exit_price - entry_price) * point_value
else:  # short
    gross_pnl = (entry_price - exit_price) * point_value

# Net PnL (after costs)
net_pnl = gross_pnl - spread_cost - commission

# R-multiple
risk = sl_distance * point_value
r_multiple = net_pnl / risk
```

---

## Validation Metrics

### Trading Metrics (Not Just Accuracy)

**Metrics Calculated**:
```python
num_trades        # Total trades on test set
win_rate          # Wins / Total
profit_factor     # Gross wins / Gross losses
avg_win           # Average winning trade $
avg_loss          # Average losing trade $
max_drawdown      # Largest peak-to-trough decline $
expectancy        # Average R-multiple per trade
sharpe            # Mean PnL / Std PnL (risk-adjusted)
total_pnl         # Cumulative profit/loss $
```

### Acceptance Thresholds

**Model must pass ALL thresholds to be saved**:

```python
MIN_TRADES_TEST = 200       # At least 200 trades
MIN_WIN_RATE = 0.48         # Win 48%+ of trades
MIN_PROFIT_FACTOR = 1.3     # Win $1.30 for every $1.00 lost
MAX_DRAWDOWN_PCT = 0.20     # Max 20% drawdown
MIN_EXPECTANCY = 0.1        # Average at least +0.1 R per trade
```

**Why these thresholds?**
- **200 trades**: Statistical significance
- **48% win rate**: Below 50% is okay if R:R is good (1.67:1)
- **1.3 PF**: Profitable after costs
- **20% DD**: Prop-firm survivable
- **0.1 R**: Positive edge per trade

### Example Pass/Fail

**PASSED**:
```
Symbol: XAUUSD, Timeframe: 15T
Trades: 347
Win Rate: 52.3%
Profit Factor: 1.68
Max Drawdown: $3,245 (13%)
Expectancy: +0.35 R
Total PnL: +$4,523
```

**FAILED (low win rate)**:
```
Symbol: XAUUSD, Timeframe: 5T
Trades: 512
Win Rate: 41.2%  ❌ < 48%
Profit Factor: 1.12 ❌ < 1.3
Expectancy: +0.05  ❌ < 0.1
```

**REJECTED** - Model NOT saved.

---

## Output Files

### Saved Models

**Format**: Pickle file containing:
```python
{
    'model': XGBClassifier,        # Trained model
    'features': list,              # Feature column names (in order)
    'symbol': str,                 # XAUUSD or XAGUSD
    'timeframe': str,              # 5T, 15T, 30T, 1H
    'metrics': dict,               # All trading metrics
    'test_accuracy': float,        # Classification accuracy
    'val_accuracy': float,
    'val_auc': float,
    'trained_at': str,             # ISO timestamp
    'train_period': str,           # Date range
    'test_period': str
}
```

**Load Model**:
```python
import joblib

model_data = joblib.load('models_citadel/XAUUSD/XAUUSD_15T.pkl')

print(f"Symbol: {model_data['symbol']}")
print(f"Features: {len(model_data['features'])}")
print(f"Win rate: {model_data['metrics']['win_rate']:.1%}")
print(f"Profit factor: {model_data['metrics']['profit_factor']:.2f}")
```

### Training Summary

**CSV file**: `models_citadel/training_summary.csv`

```csv
symbol,timeframe,status,num_trades,win_rate,profit_factor,expectancy,test_accuracy
XAUUSD,5T,FAILED,0,0.0,0.0,0.0,0.0
XAUUSD,15T,PASSED,347,0.523,1.68,0.35,0.567
XAUUSD,30T,PASSED,189,0.492,1.45,0.22,0.554
XAUUSD,1H,PASSED,124,0.508,1.51,0.28,0.561
XAGUSD,5T,PASSED,431,0.511,1.42,0.25,0.549
XAGUSD,15T,PASSED,276,0.543,1.73,0.41,0.578
XAGUSD,30T,PASSED,201,0.502,1.38,0.18,0.552
XAGUSD,1H,FAILED,87,0.425,1.08,-0.02,0.521
```

---

## Customization

### Adjust Thresholds

Edit constants in `citadel_training_system.py`:

```python
# Make it easier to pass (more permissive)
MIN_TRADES_TEST = 150
MIN_WIN_RATE = 0.45
MIN_PROFIT_FACTOR = 1.2
MAX_DRAWDOWN_PCT = 0.25
MIN_EXPECTANCY = 0.05

# Make it stricter (prop-firm grade)
MIN_TRADES_TEST = 300
MIN_WIN_RATE = 0.50
MIN_PROFIT_FACTOR = 1.5
MAX_DRAWDOWN_PCT = 0.15
MIN_EXPECTANCY = 0.15
```

### Adjust TP/SL

```python
# More conservative (tighter stops)
TP_ATR_MULTIPLE = 2.0
SL_ATR_MULTIPLE = 1.0  # R:R = 2:1

# More aggressive (wider targets)
TP_ATR_MULTIPLE = 3.0
SL_ATR_MULTIPLE = 2.0  # R:R = 1.5:1
```

### Change Label Thresholds

```python
# Require larger moves to label as LONG/SHORT
LONG_THRESHOLD_ATR = 1.0   # Was 0.8
SHORT_THRESHOLD_ATR = 1.0

# More sensitive (label smaller moves)
LONG_THRESHOLD_ATR = 0.5
SHORT_THRESHOLD_ATR = 0.5
```

---

## Integration with Existing System

### Use Citadel Models in Production

**Option 1**: Load and predict directly:
```python
import joblib
import pandas as pd

# Load model
model_data = joblib.load('models_citadel/XAUUSD/XAUUSD_15T.pkl')
model = model_data['model']
features = model_data['features']

# Prepare features (using same build_features function)
df = build_features(price_df, 'XAUUSD', '15T')

# Get latest bar features
latest = df.iloc[-1][features].values.reshape(1, -1)

# Predict
pred = model.predict(latest)
proba = model.predict_proba(latest)

# pred: 0=FLAT, 1=LONG, 2=SHORT
# proba: [flat_prob, long_prob, short_prob]
```

**Option 2**: Backtest with run_model_backtest.py:
```python
# Already compatible! Models saved with same format as train_model_temporal.py
python run_model_backtest.py --symbol XAUUSD --timeframe 15T
```

---

## Troubleshooting

### "Data not found"

**Error**:
```
FileNotFoundError: Data not found for XAUUSD at feature_store/XAUUSD/XAUUSD_5T.parquet
```

**Solution**:
1. Ensure data exists in `feature_store/{SYMBOL}/`
2. Create 5T data first (finest granularity)
3. System will resample to other timeframes automatically

### "All models failed validation"

**Possible reasons**:
1. **Too few trades**: Data period too short
2. **Poor features**: Check if features make sense
3. **Thresholds too strict**: Adjust MIN_WIN_RATE, MIN_PROFIT_FACTOR, etc.
4. **Market not tradable**: Some periods are genuinely unpredictable

**Debug**:
- Check label distribution (should be balanced)
- Check test set size (need enough bars for 200+ trades)
- Try with longer data period (5+ years recommended)

### "Model overfits (train >> test accuracy)"

**Symptoms**:
- Train accuracy: 75%
- Test accuracy: 52%

**Solutions**:
- Increase `min_child_weight` (fewer samples per leaf)
- Increase `gamma` (more regularization)
- Decrease `max_depth` (simpler trees)
- Reduce `n_estimators` (stop earlier)

---

## Best Practices

### 1. Use Long Data Periods

- Minimum: 3 years
- Recommended: 5-7 years
- Ideal: 10+ years

**Why**: Captures multiple market regimes (trending, ranging, high vol, low vol)

### 2. Retrain Regularly

Markets evolve. Retrain:
- Monthly: For active strategies
- Quarterly: For stable strategies
- After major market changes

### 3. Walk-Forward Validation

Don't just train once. Use expanding window:

```python
# Train on 2018-2022, test on 2023
# Train on 2018-2023, test on 2024
# Train on 2018-2024, test on 2025
```

This ensures model generalizes across time.

### 4. Monitor Live vs Backtest

Track:
- Live win rate vs test win rate
- Live PF vs test PF
- If diverges > 10%, investigate

### 5. Combine with Risk Management

Models are tools, not holy grails. Use:
- Position sizing (% of capital)
- Max daily loss limits
- Max correlation between positions
- Diversification across timeframes

---

## Performance Expectations

### Realistic Targets

**Good Model** (tradable):
- Win rate: 48-55%
- Profit factor: 1.3-1.7
- Expectancy: 0.1-0.3 R
- Max DD: 10-20%

**Excellent Model** (prop-firm grade):
- Win rate: 55-65%
- Profit factor: 1.7-2.5
- Expectancy: 0.3-0.6 R
- Max DD: 5-15%

**Exceptional Model** (rare):
- Win rate: 65%+
- Profit factor: 2.5+
- Expectancy: 0.6+ R
- Max DD: <10%

### What to Avoid

**Red Flags**:
- Win rate > 75% (likely overfitting or data leakage)
- Profit factor > 4.0 (too good to be true)
- < 100 trades on test (not statistically significant)
- Max DD > 30% (too risky for prop firm)

---

## Next Steps

1. **Run training**:
   ```bash
   python citadel_training_system.py
   ```

2. **Review results**:
   ```bash
   cat models_citadel/training_summary.csv
   ```

3. **Backtest passed models**:
   ```bash
   python run_model_backtest.py --symbol XAUUSD --timeframe 15T
   ```

4. **Deploy to production**:
   - Use only PASSED models
   - Monitor live performance
   - Retrain monthly

---

## Support

For questions or issues:
1. Check this guide
2. Review code comments in `citadel_training_system.py`
3. Compare with existing `train_model_temporal.py`

---

**The Citadel training system is production-ready. Good luck! 🚀**
