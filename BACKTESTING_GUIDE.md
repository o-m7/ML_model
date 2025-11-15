# Prop-Firm Backtesting Guide
**Complete Guide to Backtesting Your ML Trading Models**

---

## 📋 Overview

You now have **three production-ready backtesting tools** for evaluating your ML models under realistic prop-firm constraints:

| Tool | Purpose | Input | Use Case |
|------|---------|-------|----------|
| `backtest_propfirm.py` | Replay completed trades | CSV of trades | Validate past live trading results |
| `backtest_model.py` | Bar-by-bar simulation engine | Price data + model | Core backtesting infrastructure |
| `run_model_backtest.py` | Integrated model runner | Symbol + timeframe | **Primary tool for backtesting your models** |

---

## 🎯 Prop-Firm Challenge Rules

All backtests enforce realistic funded account rules:

```
Initial Balance:   $25,000
Profit Target:     +6% ($1,500 → $26,500)
Max Drawdown:      -4% ($1,000 → $24,000 hard stop)
Risk per Trade:    0.5% of equity
```

**Pass Conditions:**
- Reach $26,500 WITHOUT violating max drawdown ✅

**Fail Conditions:**
- Equity drops to $24,000 or below ❌
- Unable to reach profit target with available data ⚠️

---

## 🚀 Quick Start - Backtest Your Models

### Option 1: Integrated Backtest (Recommended)

Use `run_model_backtest.py` to backtest your actual trained models:

```bash
# Backtest XAUUSD 15T model (best performer - 75.9% win rate)
python run_model_backtest.py --symbol XAUUSD --timeframe 15T

# Backtest XAGUSD 30T model (top performer - 88.9% win rate)
python run_model_backtest.py --symbol XAGUSD --timeframe 30T

# Backtest with custom settings
python run_model_backtest.py \
    --symbol XAUUSD \
    --timeframe 5T \
    --balance 50000 \
    --profit-target 0.08 \
    --max-drawdown 0.05 \
    --risk-per-trade 0.01
```

**Output:**
- `backtest_results/{SYMBOL}_{TF}_trades_{timestamp}.csv` - Trade-by-trade results
- `backtest_results/{SYMBOL}_{TF}_equity_{timestamp}.csv` - Equity curve
- `backtest_results/{SYMBOL}_{TF}_summary_{timestamp}.csv` - Performance summary

### Option 2: Standalone Engine

Use `backtest_model.py` directly with custom prediction logic:

```python
from backtest_model import PropFirmBacktester
import pandas as pd

# Load your price data
price_df = pd.read_parquet("feature_store/XAUUSD/XAUUSD_15T.parquet")
price_df['symbol'] = 'XAUUSD'

# Define your prediction function
def my_predict_function(row: pd.Series) -> int:
    """
    row: Current bar with [symbol, open, high, low, close, volume]
    Returns: +1 (long), -1 (short), 0 (flat)
    """
    # Your model logic here
    # Example: Load features and predict
    return signal  # +1, -1, or 0

# Run backtest
backtester = PropFirmBacktester(
    price_df=price_df,
    predict_function=my_predict_function,
    verbose=True
)

results = backtester.run()

print(f"Status: {'PASSED' if results.passed else 'FAILED'}")
print(f"Final Equity: ${results.final_equity:,.2f}")
print(f"Win Rate: {results.win_rate:.1f}%")
```

### Option 3: Replay Past Trades

Use `backtest_propfirm.py` to validate historical trading:

```bash
# Create CSV with columns: timestamp, symbol, side, entry_price, exit_price, position_size
python backtest_propfirm.py --input my_trades.csv
```

---

## 📊 Understanding the Output

### Console Output

During backtest, you'll see real-time trade execution:

```
================================================================================
PROP-FIRM MODEL BACKTESTER
================================================================================
Initial Balance:   $25,000.00
Profit Target:     6.0% ($1,500.00)
Target Equity:     $26,500.00
Max Drawdown:      4.0% ($1,000.00)
Hard Stop:         $24,000.00
Risk per Trade:    0.50%
Data Bars:         8,760
Date Range:        2024-01-01 to 2024-12-31
================================================================================

📈 OPEN LONG: XAUUSD @ 2680.25000 | 4.17 lots | Equity: $25,000.00
✅ CLOSE LONG: XAUUSD @ 2695.80000 | PnL: $647.94 | Equity: $25,647.94 | DD: $0.00 (0.00%)
📈 OPEN LONG: XAUUSD @ 2701.50000 | 4.18 lots | Equity: $25,647.94
❌ CLOSE LONG: XAUUSD @ 2692.30000 | PnL: $-384.32 | Equity: $25,263.62 | DD: $384.32 (1.50%)

================================================================================
✅ CHALLENGE PASSED: Equity $26,523.45 >= $26,500.00
================================================================================

BACKTEST SUMMARY
================================================================================
Status:              ✅ PASSED
Initial Balance:     $25,000.00
Final Equity:        $26,523.45
Total PnL:           +$1,523.45 (+6.09%)
Peak Equity:         $26,523.45
Max Drawdown:        $412.58 (1.56%)

Total Trades:        47
Wins:                32 (68.1%)
Losses:              15
Average Win:         $+127.34
Average Loss:        $-82.19
Profit Factor:       3.24

Total Costs:         $1,892.34
================================================================================
```

### CSV Outputs

**Trades CSV:**
| symbol | direction | entry_time | exit_time | entry_price | exit_price | lots | net_pnl | equity_after |
|--------|-----------|------------|-----------|-------------|------------|------|---------|--------------|
| XAUUSD | LONG | 2024-01-01 09:00 | 2024-01-01 14:00 | 2680.25 | 2695.80 | 4.17 | 647.94 | 25647.94 |
| XAUUSD | SHORT | 2024-01-01 15:00 | 2024-01-01 18:00 | 2701.50 | 2692.30 | 4.18 | 382.64 | 26030.58 |

**Equity Curve CSV:**
| timestamp | equity | drawdown_pct | in_position |
|-----------|--------|--------------|-------------|
| 2024-01-01 09:00 | 25000.00 | 0.00 | True |
| 2024-01-01 14:00 | 25647.94 | 0.00 | False |
| 2024-01-01 15:00 | 25647.94 | 0.00 | True |

**Summary CSV:**
| metric | value |
|--------|-------|
| Status | PASSED |
| Total PnL | $+1,523.45 |
| Win Rate % | 68.1% |
| Profit Factor | 3.24 |

---

## 🔧 Configuration & Customization

### Adjusting Prop-Firm Rules

```bash
# $50k account with 8% target, 5% drawdown
python run_model_backtest.py \
    --symbol XAUUSD --timeframe 15T \
    --balance 50000 \
    --profit-target 0.08 \
    --max-drawdown 0.05

# More aggressive risk (1% per trade)
python run_model_backtest.py \
    --symbol XAUUSD --timeframe 15T \
    --risk-per-trade 0.01
```

### Adjusting Trading Costs

Edit the constants at the top of `backtest_model.py`:

```python
# Trading Costs
COMMISSION_PER_LOT = 7.0        # USD per 1.0 lot round-turn
SLIPPAGE_POINTS_MEAN = 0.0      # Mean slippage
SLIPPAGE_POINTS_STD = 2.0       # Std dev of slippage

# Symbol-Specific
SYMBOL_CONFIG = {
    "XAUUSD": {
        "spread_points": 20.0,   # Bid-ask spread
        "point_value": 0.10,     # USD per point per lot
        "min_lot": 0.01,
        "max_lot": 10.0,
    },
    "XAGUSD": {
        "spread_points": 2.0,
        "point_value": 0.01,
        "min_lot": 0.01,
        "max_lot": 10.0,
    },
}
```

### Position Sizing

Position sizing is risk-based using your stop distance:

```python
RISK_PER_TRADE_PCT = 0.005      # 0.5% of equity per trade
STOP_DISTANCE_POINTS = 300      # Stop loss distance for sizing

# Calculation:
# risk_amount = equity * 0.005
# lots = risk_amount / (stop_distance * point_value)
#
# Example: $25,000 equity, XAUUSD
# risk = $25,000 * 0.005 = $125
# lots = $125 / (300 points * $0.10) = 4.17 lots
```

---

## 📈 Backtesting Your Retrained Models

After retraining poor performers (XAUUSD 5T, 30T), validate improvements:

### 1. Retrain Models

```bash
./retrain_poor_performers.sh
```

### 2. Backtest Old vs New

```bash
# Backup old models first
cp models_rentec/XAUUSD/XAUUSD_5T.pkl models_rentec/XAUUSD/XAUUSD_5T_old.pkl

# Retrain
python train_model.py --symbol XAUUSD --tf 5T

# Backtest new model
python run_model_backtest.py --symbol XAUUSD --timeframe 5T
```

### 3. Compare Results

**Target Improvements:**

| Model | Current | Target |
|-------|---------|--------|
| XAUUSD 5T | 37.2% win rate | 55%+ |
| XAUUSD 30T | 35% SHORT win | 55%+ |

**Success Criteria:**
- ✅ Win rate ≥ 55%
- ✅ Profit factor ≥ 1.5
- ✅ Max drawdown ≤ 3%
- ✅ Pass prop-firm challenge

---

## 🎯 Best Practices

### 1. Always Run Backtests Before Deployment

```bash
# After ANY model change
python run_model_backtest.py --symbol XAUUSD --timeframe 15T

# Check for degradation
if win_rate < 55%:
    echo "❌ Model degraded - don't deploy!"
fi
```

### 2. Test Multiple Timeframes

```bash
# Test all timeframes for a symbol
for tf in 5T 15T 30T 1H; do
    python run_model_backtest.py --symbol XAUUSD --timeframe $tf
done
```

### 3. Validate Cost Assumptions

```bash
# Test with different spread assumptions
# Edit SYMBOL_CONFIG["XAUUSD"]["spread_points"]
# Low spread: 15 points
# Average: 20 points
# High: 30 points

# Ensure profitability at worst-case costs
```

### 4. Walk-Forward Testing

```bash
# Train on 2023 data
python train_model.py --symbol XAUUSD --tf 15T --end-date 2023-12-31

# Test on 2024 data
python run_model_backtest.py --symbol XAUUSD --timeframe 15T --lookback-days 365
```

---

## 🐛 Troubleshooting

### "Model not found"

```bash
# Check models exist
ls -lh models_rentec/XAUUSD/

# Retrain if missing
python train_model.py --symbol XAUUSD --tf 15T
```

### "No data found"

```bash
# Check feature store
ls -lh feature_store/XAUUSD/

# Download data if missing
# (use your data pipeline)
```

### "All signals are 0 (flat)"

- Model confidence threshold may be too high
- Check feature alignment in `IntegratedModelPredictor`
- Verify ensemble is loading correctly

### Import Errors

```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Check module paths
python -c "from ensemble_predictor import EnsemblePredictor; print('OK')"
```

---

## 📊 Performance Targets

Based on your analysis, target these metrics:

### Minimum Acceptable
- Win Rate: ≥ 50%
- Profit Factor: ≥ 1.2
- Max Drawdown: ≤ 4%
- Avg Win/Loss Ratio: ≥ 1.0

### Good Performance
- Win Rate: ≥ 55%
- Profit Factor: ≥ 1.5
- Max Drawdown: ≤ 3%
- Avg Win/Loss Ratio: ≥ 1.5

### Excellent Performance (like XAGUSD 15T/30T)
- Win Rate: ≥ 70%
- Profit Factor: ≥ 2.0
- Max Drawdown: ≤ 2%
- Avg Win/Loss Ratio: ≥ 2.0

---

## 🚀 Next Steps

1. **Retrain poor performers:**
   ```bash
   ./retrain_poor_performers.sh
   ```

2. **Backtest all models:**
   ```bash
   for symbol in XAUUSD XAGUSD; do
       for tf in 5T 15T 30T 1H; do
           python run_model_backtest.py --symbol $symbol --timeframe $tf
       done
   done
   ```

3. **Analyze results:**
   - Compare backtest results to live performance
   - Identify which models pass prop-firm challenge
   - Focus deployment on models with 60%+ win rates

4. **Deploy winners to production:**
   - Only deploy models that pass backtests
   - Monitor first 50 live signals closely
   - Compare live vs backtest performance

---

## 📚 Additional Resources

- `backtest_model.py` - Core engine documentation
- `run_model_backtest.py` - Integration examples
- `RETRAINING_PLAN.md` - Model improvement guide
- `FINAL_IMPLEMENTATION_SUMMARY.md` - System overview

---

**Your backtesting infrastructure is production-ready!** 🎉

Use these tools to validate every model change before deploying to live trading.
