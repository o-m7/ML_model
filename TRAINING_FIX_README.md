# Trading Model Performance Fix

## Problem Summary

Your models were generating severely biased signals:
- **686 Short signals, 0 Long signals** ❌
- Win Rate: 32.6% (terrible)
- Total Return: -21.2%
- Profit Factor: 0.49 (losing more than winning)

## Root Causes Identified

### 1. **Biased Labeling Logic**
- Previous labeling only checked if LONG TP hits (not SHORT)
- This created dataset with more short opportunities than long
- Model learned this bias

### 2. **Unbalanced Class Weights**
- Model training boosted minority class too much
- Created directional bias in predictions

### 3. **Insufficient Features**
- Limited feature set reduced model's ability to distinguish directions
- Needed more momentum, trend, and volatility indicators

## Fixes Applied

### ✅ Fix 1: Balanced Label Creation
**File: `train_xauusd_15t_now.py`, `train_all_timeframes_fixed.py`**

- Now checks BOTH long and short trade outcomes for each bar
- Labels as Long (1) only if long TP hits first
- Labels as Short (2) only if short TP hits first
- Labels as Flat (0) if no clear winner
- Tracks and reports long/short balance

### ✅ Fix 2: Improved Class Weights
**File: `train_xauusd_15t_now.py`**

- Uses balanced class weights (inverse frequency)
- Only slight boost to Flat class (1.2x) to encourage selectivity
- NO differential boosting between Long/Short to prevent bias

### ✅ Fix 3: Comprehensive Features
**File: `train_all_timeframes_fixed.py`**

Added 80+ features including:
- Multiple timeframe moving averages (5, 10, 20, 50, 100, 200)
- RSI on multiple periods (7, 14, 21)
- Bollinger Bands (10, 20, 30 periods)
- MACD and signal line
- ADX for trend strength
- Volume analysis
- Price action patterns
- Support/resistance levels
- Volatility ratios

### ✅ Fix 4: Prediction Validation
**File: `train_all_timeframes_fixed.py`**

- After training, validates model generates BOTH long and short signals
- Reports long/short ratio (should be ~1.0)
- Warns if severe bias detected

## How to Use (on your MacBook)

### Step 1: Pull the changes
```bash
git pull origin claude/fix-trading-model-performance-01HMR2zbUJUWu8GAmRhbf3vK
```

### Step 2: Train a single timeframe (quick test)
```bash
# Test with 15T first
python train_xauusd_15t_now.py
```

Expected output:
```
Label Distribution:
   Flat:  X,XXX (XX.X%)
   Long:  X,XXX (XX.X%)    <-- Should be >10%
   Short: X,XXX (XX.X%)    <-- Should be >10%

✅ Predictions well-balanced (Long/Short ratio: 0.8-1.2)
```

### Step 3: Train ALL timeframes
```bash
# Train 5T, 15T, 30T, 1H, 4H
python train_all_timeframes_fixed.py
```

This will:
- Fetch 2 years of data from Polygon
- Calculate comprehensive features
- Create balanced labels
- Train models for all timeframes
- Validate signal generation
- Save to `models_institutional/XAUUSD/`

### Step 4: Backtest the models
```bash
# Test each timeframe
python validate_backtest_with_costs.py --symbol XAUUSD --tf 5T
python validate_backtest_with_costs.py --symbol XAUUSD --tf 15T
python validate_backtest_with_costs.py --symbol XAUUSD --tf 30T
python validate_backtest_with_costs.py --symbol XAUUSD --tf 1H
python validate_backtest_with_costs.py --symbol XAUUSD --tf 4H
```

Expected improvements:
- **Win Rate**: Should be 48-55% (up from 32.6%)
- **Profit Factor**: Should be >1.2 (up from 0.49)
- **Sharpe Ratio**: Should be >0.3 (up from -0.41)
- **Long/Short balance**: Both directions should have signals

### Step 5: Test signal generation
```bash
# Generate live signals to verify both directions work
python signal_generator.py
```

Should see output like:
```
✅ XAUUSD 15T: LONG @ 2650.50 (TP: 2664.10, SL: 2642.00)
✅ XAUUSD 1H: SHORT @ 2651.20 (TP: 2632.40, SL: 2660.00)
```

### Step 6: Push results back
```bash
git add .
git commit -m "Fix: Balanced model training - resolves short-only bias"
git push origin claude/fix-trading-model-performance-01HMR2zbUJUWu8GAmRhbf3vK
```

## Expected Performance After Fix

| Metric | Before | Target |
|--------|--------|--------|
| Win Rate | 32.6% | 48-55% |
| Profit Factor | 0.49 | 1.2-1.8 |
| Sharpe Ratio | -0.41 | 0.3-0.8 |
| Max Drawdown | 21.5% | <8% |
| Total Return | -21.2% | +10-30% |
| Long Signals | 0 | ~50% of signals |
| Short Signals | 686 | ~50% of signals |

## Files Modified

1. **train_xauusd_15t_now.py** - Fixed labeling and training for 15T
2. **train_all_timeframes_fixed.py** - NEW: Comprehensive multi-timeframe training
3. **TRAINING_FIX_README.md** - This file

## Troubleshooting

### Issue: Still seeing bias in one direction
**Solution**: Check your data period. If training on strongly trending data, you may need:
- Longer training period (730+ days)
- More diverse market conditions
- Lower TP/SL ratios to capture more opportunities

### Issue: Low number of trades in backtest
**Solution**:
- Lower confidence threshold: `--confidence 0.45` (default 0.55)
- Use lower risk: `--risk 0.5` (default 1.0)
- Check if data is sufficient

### Issue: Model training fails
**Solution**:
- Ensure POLYGON_API_KEY is set in `.env`
- Check internet connection
- Verify you have 730+ days of data available

## Next Steps After Training

1. **Review model files** in `models_institutional/XAUUSD/`
2. **Run comprehensive backtests** on all timeframes
3. **Validate in signal_generator.py** that both directions work
4. **Monitor live performance** for 1-2 weeks before full deployment
5. **Re-train monthly** with latest data to adapt to market changes

## Questions?

Check the code comments in:
- `train_all_timeframes_fixed.py:139-250` - Balanced labeling logic
- `train_all_timeframes_fixed.py:533-615` - Model training with balanced weights
- `train_all_timeframes_fixed.py:618-680` - Signal generation validation

---

**CRITICAL**: After training, ALWAYS run the backtest validation to ensure:
1. Both long and short signals are generated
2. Performance metrics are acceptable
3. No severe bias exists

Good luck! 🚀
