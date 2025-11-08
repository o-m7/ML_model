# 🔄 Retraining Strategy - Two Approaches

## 📊 **Overview**

The system uses **TWO different retraining approaches** with **completely different purposes**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                       │
│   1️⃣  WEEKLY DATA REFRESH (automated_retraining.py)                  │
│      Purpose: Prevent staleness by adding new market data           │
│      Frequency: Weekly (every Sunday)                                │
│      Changes: ZERO - Only adds new data                              │
│                                                                       │
│   2️⃣  LIVE TRADE LEARNING (retrain_from_live_trades.py)              │
│      Purpose: Learn from mistakes and improve strategy               │
│      Frequency: Daily (after collecting live trades)                 │
│      Changes: YES - Adapts based on performance                      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ **WEEKLY DATA REFRESH** (`automated_retraining.py`)

### **Purpose:**
Keep models fresh with latest market data **WITHOUT changing anything about the strategy**.

### **What It Does:**
✅ Fetches ONLY new data from last 7 days  
✅ Appends to existing historical dataset  
✅ Retrains with EXACT SAME parameters:
  - Same TP/SL multipliers
  - Same confidence thresholds
  - Same features
  - Same hyperparameters
  - Same class weights

✅ Validates against SAME benchmarks  
✅ Deploys ONLY if passes ALL benchmarks  

### **What It Does NOT Do:**
❌ Does NOT change TP or SL  
❌ Does NOT change min_confidence or min_edge  
❌ Does NOT experiment with new features  
❌ Does NOT adjust based on win rate  
❌ Does NOT change training methodology  

### **Why This Matters:**
Markets evolve over time. A model trained on January-October 2025 data might not capture November patterns. Weekly data refresh ensures the model sees the latest price action **while keeping the proven strategy intact**.

### **Example:**
```python
# Week 1: Model trained on data from Jan 1 - Oct 22, 2025
# Week 2: Fetch Oct 23 - Oct 29 → Retrain with Jan 1 - Oct 29
# Week 3: Fetch Oct 30 - Nov 5  → Retrain with Jan 1 - Nov 5
# Week 4: Fetch Nov 6 - Nov 12  → Retrain with Jan 1 - Nov 12

# Strategy parameters NEVER change:
XAUUSD_5T = {
    'tp': 1.9,        # ← SAME
    'sl': 1.0,        # ← SAME
    'min_conf': 0.43, # ← SAME
    'min_edge': 0.08, # ← SAME
    'pos_size': 0.005 # ← SAME
}
```

### **Deployment Rules:**
```python
# Model is deployed ONLY if it passes ALL benchmarks:
✅ Profit Factor ≥ 1.05
✅ Max Drawdown ≤ 7.5%
✅ Sharpe Ratio ≥ 0.05
✅ Win Rate ≥ 39%
✅ Min Trades (varies by timeframe)

# If ANY benchmark fails → Model is saved as FAILED, NOT deployed
```

### **Runs:**
- **Automatically:** Every Sunday at 00:00 UTC (GitHub Actions)
- **Manually:** `python3 automated_retraining.py`

---

## 2️⃣ **LIVE TRADE LEARNING** (`retrain_from_live_trades.py`)

### **Purpose:**
**Learn from actual trading mistakes** and adapt the strategy to improve performance.

### **What It Does:**
✅ Analyzes every losing trade  
✅ Identifies error patterns:
  - Overconfident losses → Increase min_confidence
  - Directional bias → Boost underperforming class weight
  - Too many SL hits → Widen stop loss
  - Low-confidence losers → Tighten entry criteria

✅ Adjusts training to avoid repeating mistakes  
✅ Validates against benchmarks before deployment  

### **What It Changes:**
🔧 **Sample Weights** - Boost weight of losing patterns  
🔧 **Class Weights** - Balance long/short if bias detected  
🔧 **Confidence Threshold** - Increase if losers too confident  
🔧 **Training Focus** - Emphasize avoiding past mistakes  

### **Why This Matters:**
Backtests can't predict every market condition. Live trading reveals weaknesses. This system **automatically corrects** those weaknesses.

### **Example:**
```python
# PROBLEM DETECTED:
Long trades: 25 winners, 48 losers (34% WR)
Short trades: 67 winners, 12 losers (85% WR)
→ Model is BAD at predicting upward moves!

# AUTOMATIC FIX:
class_weights = {
    0: 1.0,   # Flat
    1: 1.5,   # Up ← BOOSTED from 1.0
    2: 1.0    # Down
}

# RESULT:
After retraining with boosted Up class:
Long trades: 42 winners, 38 losers (52% WR) ← IMPROVED!
Short trades: 63 winners, 15 losers (81% WR) ← MAINTAINED
```

### **Another Example:**
```python
# PROBLEM DETECTED:
Losers: Avg confidence = 0.62
Winners: Avg confidence = 0.59
→ Model is overconfident on losing trades!

# AUTOMATIC FIX:
Recommendation: Increase MIN_CONFIDENCE from 0.40 to 0.65

# RESULT:
Trade count drops from 247 to 189
But win rate increases from 48% to 56%
Profit factor increases from 1.42 to 1.87
```

### **Deployment Rules:**
```python
# Step 1: Check accuracy improvement
if new_accuracy < old_accuracy - 0.02:
    ❌ Reject (significant degradation)

# Step 2: Run FULL backtest on OOS data
new_backtest_results = backtest(new_model, test_data)

# Step 3: Validate ALL benchmarks
✅ Profit Factor ≥ 1.05
✅ Max Drawdown ≤ 7.5%
✅ Sharpe Ratio ≥ 0.05
✅ Win Rate ≥ 39%
✅ Min Trades (varies by timeframe)

# Step 4: Deploy ONLY if passes
if ALL benchmarks pass:
    ✅ Backup old model
    ✅ Deploy new model
    ✅ Update ONNX + Supabase
else:
    ❌ Keep old model (don't risk degradation)
```

### **Runs:**
- **Automatically:** Every day at 00:00 UTC (GitHub Actions)
- **Manually:** `python3 trade_collector.py && python3 retrain_from_live_trades.py`

---

## 📊 **Comparison Table**

| Feature | Weekly Data Refresh | Live Trade Learning |
|---------|-------------------|---------------------|
| **Frequency** | Weekly (Sunday) | Daily (after trades) |
| **Data Source** | Polygon (OHLCV) | Supabase (trades table) |
| **Purpose** | Prevent staleness | Learn from mistakes |
| **Changes Strategy?** | ❌ NO | ✅ YES |
| **Changes Parameters?** | ❌ NO | ✅ YES (if needed) |
| **Requires Live Trades?** | ❌ NO | ✅ YES |
| **Deployment Threshold** | Pass benchmarks | Pass benchmarks + better |
| **Risk Level** | 🟢 Low (no changes) | 🟡 Medium (strategic changes) |

---

## 🎯 **Combined Strategy**

Both approaches work together:

```
Week 1:
├─ Sunday: Weekly data refresh (add Oct 23-29 data)
├─ Monday: Live trade learning (learn from weekend trades)
├─ Tuesday: Live trade learning (learn from Monday trades)
├─ Wednesday: Live trade learning (learn from Tuesday trades)
├─ Thursday: Live trade learning (learn from Wednesday trades)
├─ Friday: Live trade learning (learn from Thursday trades)
└─ Saturday: Live trade learning (learn from Friday trades)

Week 2:
├─ Sunday: Weekly data refresh (add Oct 30-Nov 5 data) ← Fresh data
├─ Monday: Live trade learning (all improvements from Week 1) ← Smarter
└─ ... continues ...
```

**Result:**
- Models stay **fresh** (weekly data updates)
- Models get **smarter** (daily learning from mistakes)
- Models stay **safe** (strict benchmark validation)

---

## 🔒 **Benchmark Enforcement**

### **Strict Validation:**
```python
from benchmark_validator import BenchmarkValidator

# Before ANY deployment:
passes, failures = BenchmarkValidator.validate(results, timeframe, strict=True)

if passes:
    ✅ Deploy to production
    ✅ Convert to ONNX
    ✅ Upload to Supabase
    ✅ Signals go live
else:
    ❌ Save as FAILED.pkl
    ❌ Do NOT deploy
    ❌ Keep old model active
    ❌ Log failures for review
```

### **What Gets Validated:**
```python
MIN_PROFIT_FACTOR = 1.05
MAX_DRAWDOWN_PCT = 7.5
MIN_SHARPE = 0.05
MIN_WIN_RATE = 39.0
MIN_TRADES_BY_TF = {
    "5T": 200,
    "15T": 150,
    "30T": 100,
    "1H": 60,
    "4H": 40
}
```

**Every single retrained model** must pass ALL benchmarks or it will NOT be deployed.

---

## 📋 **Files Overview**

| File | Purpose | Changes Strategy? | Validates Benchmarks? |
|------|---------|-------------------|----------------------|
| `automated_retraining.py` | Weekly data refresh | ❌ NO | ✅ YES (strict) |
| `retrain_from_live_trades.py` | Learn from mistakes | ✅ YES | ✅ YES (strict) |
| `benchmark_validator.py` | Validate performance | N/A | ✅ Core validation |
| `trade_collector.py` | Fetch & analyze trades | N/A | ❌ NO |
| `trade_learning_dashboard.py` | Visualize learning | N/A | ❌ NO |

---

## 🚀 **Deployment Flow**

### **Scenario 1: Weekly Data Refresh (Sunday)**
```
1. Fetch Oct 30 - Nov 5 data from Polygon
2. Append to existing historical data
3. Retrain with SAME parameters
4. Run backtest on OOS data
5. Validate benchmarks (strict)
6. IF passes → Deploy
   ELSE → Keep old model
```

### **Scenario 2: Live Trade Learning (Daily)**
```
1. Fetch all trades from Supabase
2. Analyze losing patterns
3. Identify adjustments needed
4. Retrain with adjusted weights/samples
5. Run backtest on OOS data
6. Validate benchmarks (strict)
7. IF passes AND better → Deploy
   ELSE → Keep old model
```

### **Scenario 3: Model Fails Benchmarks**
```
1. Retrain completes
2. Backtest shows: PF=0.92 (< 1.05 required)
3. Benchmark validator: ❌ FAILS
4. Save as XAUUSD_5T_FAILED.pkl
5. Log failure reasons
6. Keep XAUUSD_5T_PRODUCTION_READY.pkl active
7. Live trading continues with OLD model
8. Alert user to investigate
```

---

## ✅ **Summary**

### **Weekly Data Refresh:**
- **What:** Add new data, no strategy changes
- **Why:** Prevent model staleness
- **Risk:** 🟢 Very low (proven strategy, just fresher data)
- **Deploys if:** Passes all benchmarks

### **Live Trade Learning:**
- **What:** Adapt strategy based on live performance
- **Why:** Learn from mistakes, improve continuously
- **Risk:** 🟡 Medium (strategic changes, but validated)
- **Deploys if:** Passes all benchmarks AND performs better

### **Both Approaches:**
- ✅ Strict benchmark validation
- ✅ Automatic backup of old models
- ✅ No deployment if benchmarks fail
- ✅ Fully automated via GitHub Actions
- ✅ Manual override available

---

**🎯 Result: A self-improving system that stays fresh AND learns from experience, while NEVER deploying underperforming models.**

