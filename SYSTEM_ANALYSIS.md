# ML Trading System Analysis & Action Plan

## 🔍 Current Issues Identified

### 1. **Training Speed (1 hour vs 5 minutes expected)**

**Root Causes:**
- **Triple-barrier labeling is O(n*horizon)**: Iterates through 40,000+ bars, each checking 40 bars ahead = 1.6M operations per symbol
- **Full correlation matrix computation**: With 243 features, creates 59K correlation pairs
- **10-fold walk-forward CV**: Each fold trains 3 models (XGB, LGBM, Linear) = 30 model fits per symbol
- **Feature importance ranking**: Additional XGBoost fit for feature selection

**Solution:**
- Vectorize triple-barrier labeling using numpy
- Cache correlation matrix between runs
- Reduce to 5 folds for faster iteration
- Skip feature importance if <100 features

### 2. **High Drawdown (98% vs 6% target)**

**Root Cause (PARTIALLY FIXED):**
- ✅ Added leverage support (50:1)
- ✅ Fixed position sizing to use margin
- ⚠️ **STILL MISSING**: The system is NOT respecting COOLDOWN_BARS properly
- ⚠️ **STILL MISSING**: No correlation control between symbols (can take 8 correlated positions)
- ⚠️ **STILL MISSING**: No global portfolio stop-loss

**Remaining Issues:**
- Models can take consecutive trades immediately after losses
- FX pairs are 70%+ correlated, multiplying losses
- No circuit breaker for portfolio-level drawdown

### 3. **Low PF/Sharpe for FX vs XAUUSD**

**Root Cause:**
- **Single strategy per timeframe**: Currently uses ONE strategy regardless of market regime
- **FX needs regime adaptation**: FX ranges 60% of time vs XAUUSD trends 45% of time
- **Wrong TP/SL ratios for FX**: XAUUSD benefits from 1.5R TP, FX needs 2.0R+ due to tighter ranges
- **No session filtering**: FX profits come from London/NY overlap, not Asia session

## 🎯 Current Strategy Assignments

| Timeframe | Current Strategy | Type | Issue |
|-----------|-----------------|------|-------|
| 5m | Momentum Breakout | Trend | Fails in ranging FX markets |
| 15m | Mean-Revert VWAP | Range | Fails in trending XAUUSD |
| 30m | Pullback-to-Trend | Trend | No regime detection |
| 1h | Breakout + Retest | Trend | No regime detection |
| 2h | Momentum ADX+ATR | Trend | No regime detection |
| 4h | MTF Alignment | Trend | No regime detection |

**Problem**: Currently uses FIXED strategy per timeframe. Needs REGIME-BASED selection.

## 🔧 Action Plan

### Phase 1: Fix Immediate Issues (Today)

1. **Vectorize triple-barrier labeling** (10x speedup)
2. **Add portfolio-level risk controls**:
   - Max 3 concurrent positions
   - Max 2% total portfolio risk
   - 6% circuit breaker drawdown stop
3. **Symbol-specific TP/SL ratios**:
   - XAUUSD/XAGUSD: 1.5R TP, 1.0R SL (current)
   - FX pairs: 2.0R TP, 0.8R SL (better R:R)
4. **Proper cooldown enforcement** (currently broken)

### Phase 2: Regime-Based Multi-Strategy (Next)

1. **Add regime detection module**:
   - Compute: ADX(14), ATR percentile, MA slope, BB width
   - Output: "TREND" or "RANGE" signal

2. **Create dual strategies per timeframe**:
   - Keep existing as "Trend Strategy"
   - Add complementary "Range Strategy"
   - Runtime: Select based on regime

3. **Symbol-specific tuning**:
   - XAUUSD: Favor trend strategies (45% trend regime)
   - EURUSD: Favor range strategies (65% range regime)
   - Session filters for FX (London/NY)

### Phase 3: Portfolio Orchestration (Later)

1. **Correlation-aware position limits**
2. **Dynamic allocation based on regime confidence**
3. **Real-time regime monitoring**

## 📋 Retrain Commands (Symbol by Symbol)

### XAUUSD (Start Here - Best Performer)
```bash
# All timeframes one-by-one
python production_training_system.py --symbol XAUUSD --tf 5T
python production_training_system.py --symbol XAUUSD --tf 15T
python production_training_system.py --symbol XAUUSD --tf 30T
python production_training_system.py --symbol XAUUSD --tf 1H
python production_training_system.py --symbol XAUUSD --tf 2H
python production_training_system.py --symbol XAUUSD --tf 4H
```

### XAGUSD
```bash
python production_training_system.py --symbol XAGUSD --tf 5T
python production_training_system.py --symbol XAGUSD --tf 15T
python production_training_system.py --symbol XAGUSD --tf 30T
python production_training_system.py --symbol XAGUSD --tf 1H
python production_training_system.py --symbol XAGUSD --tf 2H
python production_training_system.py --symbol XAGUSD --tf 4H
```

### EURUSD
```bash
python production_training_system.py --symbol EURUSD --tf 5T
python production_training_system.py --symbol EURUSD --tf 15T
python production_training_system.py --symbol EURUSD --tf 30T
python production_training_system.py --symbol EURUSD --tf 1H
python production_training_system.py --symbol EURUSD --tf 2H
python production_training_system.py --symbol EURUSD --tf 4H
```

### GBPUSD
```bash
python production_training_system.py --symbol GBPUSD --tf 5T
python production_training_system.py --symbol GBPUSD --tf 15T
python production_training_system.py --symbol GBPUSD --tf 30T
python production_training_system.py --symbol GBPUSD --tf 1H
python production_training_system.py --symbol GBPUSD --tf 2H
python production_training_system.py --symbol GBPUSD --tf 4H
```

### AUDUSD
```bash
python production_training_system.py --symbol AUDUSD --tf 5T
python production_training_system.py --symbol AUDUSD --tf 15T
python production_training_system.py --symbol AUDUSD --tf 30T
python production_training_system.py --symbol AUDUSD --tf 1H
python production_training_system.py --symbol AUDUSD --tf 2H
python production_training_system.py --symbol AUDUSD --tf 4H
```

### NZDUSD
```bash
python production_training_system.py --symbol NZDUSD --tf 5T
python production_training_system.py --symbol NZDUSD --tf 15T
python production_training_system.py --symbol NZDUSD --tf 30T
python production_training_system.py --symbol NZDUSD --tf 1H
python production_training_system.py --symbol NZDUSD --tf 2H
python production_training_system.py --symbol NZDUSD --tf 4H
```

### USDJPY
```bash
python production_training_system.py --symbol USDJPY --tf 5T
python production_training_system.py --symbol USDJPY --tf 15T
python production_training_system.py --symbol USDJPY --tf 30T
python production_training_system.py --symbol USDJPY --tf 1H
python production_training_system.py --symbol USDJPY --tf 2H
python production_training_system.py --symbol USDJPY --tf 4H
```

### USDCAD
```bash
python production_training_system.py --symbol USDCAD --tf 5T
python production_training_system.py --symbol USDCAD --tf 15T
python production_training_system.py --symbol USDCAD --tf 30T
python production_training_system.py --symbol USDCAD --tf 1H
python production_training_system.py --symbol USDCAD --tf 2H
python production_training_system.py --symbol USDCAD --tf 4H
```

## 🎲 Expected Training Time (After Optimization)

- **Current**: ~60 minutes per symbol/timeframe
- **After vectorization**: ~10-15 minutes per symbol/timeframe
- **Total for all 48 combinations**: ~8-12 hours

## ⚠️ Critical Fixes Needed BEFORE Mass Retraining

1. **Vectorize triple-barrier labeling** (10x speedup)
2. **Symbol-specific TP/SL ratios** (improve FX performance)
3. **Proper cooldown enforcement** (reduce overtrading)
4. **Portfolio risk limits** (prevent >6% DD)

**Recommendation**: Fix these 4 issues FIRST, then retrain XAUUSD 1H as test, THEN proceed with full retraining.

