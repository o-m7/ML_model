# FORENSIC AUDIT REPORT
## Institutional ML Trading System - Failure Analysis

### EXECUTIVE SUMMARY
All 22 walk-forward segments failed. Average PF: 0.97, Average Sharpe: 0.15.
No segments met viability criteria (PF≥1.5, Sharpe≥0.5).

---

## ROOT CAUSES IDENTIFIED

### 1. MODEL FAILURE - No Predictive Power (CRITICAL)
**Evidence:**
- XGBoost Validation AUC: 0.526 (random = 0.5)
- LightGBM Validation AUC: 0.524 (random = 0.5)
- MLP Validation AUC: 0.500 (exactly random!)

**Diagnosis:**
- 74 features contain no predictive signal for TP/SL-based labels
- Models are overfitting noise (train AUC 0.63-0.85, val AUC ~0.5)
- Label creation might be flawed or features misaligned

**Impact:**
With random predictions (50/50), strategy cannot be profitable after costs.

---

### 2. EQUITY WIPEOUT - Cascading Failure (CRITICAL)
**Evidence:**
```
Segment 1:  $25,000 → $13,375 (-46.5%)
Segment 8:  → $2,250 (-91.0%)
Segment 10+: $0 equity → 0 position size → no trades possible
```

**Diagnosis:**
- Poor model leads to net losses
- Losses compound across segments
- Once equity hits ~$0, all future segments start with $0
- No equity floor or reset mechanism

**Impact:**
Later segments cannot trade (position_size = 0.0000 lots).

---

### 3. BACKTEST EXECUTION - Multiple Issues

**Issue 3a: No Equity Protection**
- System allows equity to drop to $0
- Should reset or stop trading below minimum

**Issue 3b: Position Sizing with $0**
- `risk_amount = equity * 0.01 = $0 * 0.01 = $0`
- `position_size = $0 / (sl_distance * 100) = 0 lots`
- System continues "trading" with 0 lots

**Issue 3c: Excessive Costs**
- Spread: $0.30/oz + slippage 0.01%
- With 90% signal rate, costs dominate P&L
- Need to reduce signal frequency

---

### 4. LABEL ENGINEERING - Potential Mismatch

**Current Approach:**
- TP/SL-based labels (TP=2.0×ATR, SL=1.0×ATR)
- Look forward 8 bars
- Label = 1 if TP hit before SL AND R≥0.3

**Issues:**
- 34.8% LONG opportunities (imbalanced)
- Features might not predict TP/SL outcomes
- Could be lookahead bias or noise

---

### 5. FEATURE QUALITY - No Signal

**Possible Causes:**
- Technical indicators lag price (already priced in)
- Quote features not predictive for 15min bars
- Cross-asset correlation weak
- Need microstructure/order flow features

---

## FIXES REQUIRED (Priority Order)

### Fix #1: Add Equity Protection (IMMEDIATE)
```python
# In WalkForwardValidator.validate():
MIN_EQUITY = 5000  # Minimum to continue trading

if self.cumulative_equity < MIN_EQUITY:
    print(f"⚠️  Equity below minimum (${self.cumulative_equity:.2f})")
    print(f"   Resetting to ${self.config.initial_capital:.2f}")
    self.cumulative_equity = self.config.initial_capital
```

### Fix #2: Improve Model Training (CRITICAL)
```python
# Reduce overfitting:
xgb_params = {
    'max_depth': 3,  # Was 5
    'learning_rate': 0.01,  # Was 0.05
    'n_estimators': 50,  # Was 100
    'min_child_weight': 10,  # Add regularization
}

# Add early stopping:
eval_set = [(X_val, y_val)]
model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10)
```

### Fix #3: Simplify Labels (URGENT)
```python
# Try simpler forward return labels:
df['label'] = (df['close'].shift(-5) > df['close'] * 1.001).astype(int)

# Or directional labels:
df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
```

### Fix #4: Increase Signal Selectivity
```python
# Raise threshold to reduce trade frequency:
fixed_threshold: float = 0.60  # Was 0.50

# Or use top quantile:
use_fixed_threshold: bool = False
signal_quantile: float = 0.90  # Top 10% only
```

### Fix #5: Feature Engineering Review
```python
# Focus on features that SHOULD predict:
- Recent momentum (1-3 bars)
- Volume spikes
- Volatility regime changes
- Quote imbalance (if available)

# Remove lagging indicators:
- Long-term MAs (50, 100, 200)
- Slow oscillators
- Correlation features
```

---

## RECOMMENDED ACTION PLAN

**Phase 1: Emergency Fixes (Do First)**
1. Pull latest code with equity safety checks
2. Add equity floor/reset mechanism
3. Increase signal threshold to 0.65
4. Reduce max_position_size to 0.5 lots

**Phase 2: Model Improvements**
1. Simplify labels to forward returns
2. Reduce model complexity (prevent overfitting)
3. Add early stopping
4. Feature selection (keep only top 20 features)

**Phase 3: Strategy Refinement**
1. Test on single best-performing segment
2. Tune hyperparameters on that segment
3. Validate on out-of-sample data
4. Only then run full walk-forward

---

## EXPECTED OUTCOMES AFTER FIXES

**Realistic Targets:**
- Average PF: 1.0 - 1.2 (breakeven to slight profit)
- Average Sharpe: -0.2 to 0.3 (barely positive)
- Win Rate: 48-52% (near random)
- Viable Segments: 2-5 out of 22 (10-20%)

**Why Low Expectations?**
- XAUUSD is highly efficient market
- 15-minute bars have little edge
- Costs are significant relative to edge
- This is institutional-grade reality

---

## CONCLUSION

The system is technically sound but economically failing due to:
1. Zero predictive signal in features/labels
2. No protection against equity wipeout
3. Overfitting noise instead of signal

**Next Steps:** Implement Phase 1 fixes immediately, then reassess.
