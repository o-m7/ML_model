# 🚀 XAUUSD LIVE TRADING RUNBOOK

**Version:** 2.0 (Post-Fix)
**Last Updated:** 2025-11-13
**Maintainer:** Quant/SRE Team

---

## 📋 TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Procedure](#deployment-procedure)
4. [Monitoring & Alerts](#monitoring--alerts)
5. [Troubleshooting](#troubleshooting)
6. [Rollback Procedure](#rollback-procedure)
7. [Performance Targets](#performance-targets)
8. [Configuration Reference](#configuration-reference)

---

## 🎯 SYSTEM OVERVIEW

### Architecture
```
┌─────────────────┐
│  Polygon API    │ ← Live market data (5T, 15T, 30T, 1H bars)
└────────┬────────┘
         │
┌────────▼────────────────────────────────────────┐
│  Live Trading Engine (live_trading_engine.py)  │
│  - Fetches OHLCV data                          │
│  - Builds features (live_feature_utils)        │
│  - Applies execution guardrails                │
│  - Generates signals                            │
└────────┬────────────────────────────────────────┘
         │
┌────────▼────────┐         ┌──────────────────┐
│  Market Costs   │◄────────┤  Guardrails      │
│  (market_costs) │         │  (execution_     │
│  - TP/SL params │         │   guardrails)    │
│  - Spread/costs │         │  - Staleness     │
└────────┬────────┘         │  - Spread filter │
         │                  │  - Session filter│
┌────────▼────────┐         │  - Vol clamp     │
│  Model Ensemble │         └──────────────────┘
│  (XAUUSD_15T_   │
│   PRODUCTION.   │
│   pkl)          │
└────────┬────────┘
         │
┌────────▼────────┐
│  Supabase DB    │ ← Stores signals, tracks performance
│  (live_signals) │
└─────────────────┘
```

### Critical Files
```
market_costs.py                    # TP/SL, costs - SINGLE SOURCE OF TRUTH
execution_guardrails.py            # Filters for live execution
live_feature_utils.py              # Feature calculation (used everywhere)
live_trading_engine_fixed.py       # Live execution (POST-FIX)
signal_generator.py                # Signal generation
models_production/XAUUSD/*.pkl     # Trained models
config.yaml                        # High-level config (DEPRECATED for TP/SL)
```

---

## ✅ PRE-DEPLOYMENT CHECKLIST

### 1. Environment Check
```bash
# Verify Python packages
pip install -r requirements.txt

# Required packages:
# - pandas >= 2.3.0
# - numpy >= 2.3.0
# - pandas_ta
# - xgboost
# - lightgbm
# - scikit-learn
# - supabase-py
# - python-dotenv

# Verify environment variables
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

required = ['POLYGON_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']
for var in required:
    assert os.getenv(var), f'Missing {var}'
print('✅ All env vars present')
"
```

### 2. Model Validation
```bash
# Check production models exist
ls -lh models_production/XAUUSD/XAUUSD_*_PRODUCTION_READY.pkl

# Expected: 5 files (5T, 15T, 30T, 1H, 4H)

# Validate model can load
python -c "
import pickle
from pathlib import Path

model_path = Path('models_production/XAUUSD/XAUUSD_15T_PRODUCTION_READY.pkl')
with open(model_path, 'rb') as f:
    system = pickle.load(f)

print(f'✅ Model loaded: {model_path.name}')
print(f'   Backtest Sharpe: {system[\"results\"][\"backtest\"][\"sharpe\"]:.2f}')
print(f'   Backtest PF: {system[\"results\"][\"backtest\"][\"profit_factor\"]:.2f}')
"
```

### 3. Cost & Guardrail Validation
```bash
# Test market costs module
python market_costs.py

# Expected output: ✅ All X symbols validated

# Test execution guardrails
python execution_guardrails.py

# Expected output: Guardrail test results
```

### 4. Feature Parity Test
```bash
# Run feature parity test (if exists)
python test_feature_parity.py

# This validates that features calculated in live match training
```

### 5. Backtest Validation
```bash
# Run backtest with POST-FIX costs and logic
python true_backtest_engine.py --symbol XAUUSD --tf 15T --use-market-costs

# Expected metrics (after fixes):
# - Win Rate: 50-60%
# - Profit Factor: 1.3-1.8
# - Sharpe/trade: 0.20-0.40
# - Max DD: < 6%
```

### 6. Paper Trading Validation
```bash
# Run in paper trading mode for 1 week BEFORE going live
python live_trading_engine_fixed.py --symbol XAUUSD --tf 15T --paper-mode

# Monitor daily:
# - Win rate vs backtest
# - Avg R vs backtest
# - Execution latency < 250ms
# - Guardrail rejection rate (~30-50% is healthy)
```

---

## 🚀 DEPLOYMENT PROCEDURE

### Step 1: Deploy to Paper Trading (Week 1)
```bash
# Set risk to ZERO (paper only)
export RISK_PER_TRADE_PCT=0.0
export PAPER_MODE=true

# Start live engine
python live_trading_engine_fixed.py \
    --symbol XAUUSD \
    --timeframe 15T \
    --paper-mode \
    --log-level INFO \
    > logs/paper_trading_$(date +%Y%m%d).log 2>&1 &

# Monitor logs
tail -f logs/paper_trading_$(date +%Y%m%d).log
```

**Monitor daily:**
- [ ] Win rate within 5% of backtest
- [ ] Avg R within 0.05R of backtest
- [ ] No execution errors
- [ ] Latency < 250ms
- [ ] Guardrail filters working

**Exit Criteria:**
- 20+ paper signals generated
- Paper WR ≥ 50%
- Paper PF ≥ 1.2
- No critical bugs

---

### Step 2: Deploy to Live (0.1% Risk)
```bash
# FIRST LIVE DEPLOYMENT - Conservative risk!
export RISK_PER_TRADE_PCT=0.001  # 0.1% risk (10x lower than target)
export PAPER_MODE=false

# Start live engine with MINIMAL risk
python live_trading_engine_fixed.py \
    --symbol XAUUSD \
    --timeframe 15T \
    --risk-pct 0.001 \
    --max-daily-trades 2 \
    --log-level INFO \
    > logs/live_trading_$(date +%Y%m%d).log 2>&1 &

# Get process ID
echo $! > logs/live_engine.pid
```

**Monitor HOURLY for first 48 hours:**
- [ ] Trades execute correctly
- [ ] TP/SL hit as expected
- [ ] No slippage anomalies
- [ ] Broker confirms match our records

---

### Step 3: Scale Up Risk (Gradual)
```bash
# After 10 successful trades at 0.1% risk:

# Week 2: Increase to 0.25% risk
export RISK_PER_TRADE_PCT=0.0025

# Week 3: Increase to 0.5% risk
export RISK_PER_TRADE_PCT=0.005

# Week 4+: Full 1% risk (if metrics hold)
export RISK_PER_TRADE_PCT=0.01
```

**Scale-Up Criteria:**
- ✅ No execution errors in previous week
- ✅ Win rate within 10% of backtest
- ✅ Profit factor ≥ 1.2
- ✅ Max drawdown < 3%

**STOP SCALING IF:**
- ❌ Win rate drops below 40%
- ❌ Profit factor < 1.0
- ❌ Drawdown > 5%
- ❌ Execution errors or latency spikes

---

## 📊 MONITORING & ALERTS

### Real-Time Monitoring
```bash
# Check live status
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Get recent signals
signals = supabase.table('live_signals').select('*').order('timestamp', desc=True).limit(10).execute()

print(f'Recent signals: {len(signals.data)}')
for sig in signals.data[:5]:
    print(f\"  {sig['timestamp']}: {sig['symbol']} {sig['signal_type']} @ {sig['confidence']:.2%}\")
"
```

### Performance Metrics (Daily)
```bash
# Generate daily report
python -c "
import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Get last 24h signals
signals = supabase.table('live_signals').select('*').gte(
    'timestamp', pd.Timestamp.now() - pd.Timedelta(days=1)
).execute()

df = pd.DataFrame(signals.data)
if len(df) > 0:
    print(f'Last 24h: {len(df)} signals')
    print(f'  Long: {(df[\"signal_type\"] == \"long\").sum()}')
    print(f'  Short: {(df[\"signal_type\"] == \"short\").sum()}')
    print(f'  Avg confidence: {df[\"confidence\"].mean():.2%}')
else:
    print('No signals in last 24h')
"
```

### Alert Thresholds
Set up alerts (email/SMS) for:
- ❌ Win rate < 40% over rolling 20 trades
- ❌ Drawdown > 5%
- ❌ 3+ consecutive losses
- ❌ Execution latency > 500ms
- ❌ No signals generated for 24+ hours (data feed issue)
- ❌ Guardrail rejection rate > 80% (filters too strict)

---

## 🔧 TROUBLESHOOTING

### Issue: No Signals Being Generated

**Check:**
1. Is data feed working?
   ```bash
   python -c "
   from live_trading_engine_fixed import fetch_polygon_data
   df = fetch_polygon_data('XAUUSD', 15, 50)
   print(f'Got {len(df)} bars' if df is not None else 'NO DATA')
   "
   ```

2. Are guardrails blocking everything?
   ```bash
   # Check guardrail logs
   grep "FAIL" logs/live_trading_*.log | tail -20
   ```

3. Is model loading correctly?
   ```bash
   python -c "
   import pickle
   model = pickle.load(open('models_production/XAUUSD/XAUUSD_15T_PRODUCTION_READY.pkl', 'rb'))
   print('Model loaded OK')
   "
   ```

**Fix:**
- Check `POLYGON_API_KEY` is valid and has quota
- Relax guardrails temporarily (use `get_aggressive_guardrails()`)
- Verify model file not corrupted

---

### Issue: Win Rate Much Lower Than Backtest

**Check:**
1. Are costs being applied correctly?
   ```bash
   python market_costs.py  # Should show cost calculations
   ```

2. Are TP/SL params aligned?
   ```bash
   python -c "
   from market_costs import get_tp_sl
   params = get_tp_sl('XAUUSD', '15T')
   print(f'TP: {params.tp_atr_mult}R, SL: {params.sl_atr_mult}R')
   print(f'R:R ratio: {params.risk_reward_ratio:.2f}:1')
   "
   ```

3. Are features calculated correctly?
   ```bash
   # Run feature parity test
   python test_feature_parity.py
   ```

**Fix:**
- Verify `market_costs.py` is being imported correctly
- Check logs for feature calculation warnings
- Compare live fills vs expected prices (slippage check)

---

### Issue: High Latency (>500ms)

**Check:**
1. Network latency to Polygon API
   ```bash
   curl -w "@curl-format.txt" -o /dev/null -s "https://api.polygon.io/v2/aggs/ticker/C:XAUUSD/range/1/minute/2025-01-01/2025-01-10?apiKey=$POLYGON_API_KEY"
   ```

2. Feature calculation time
   ```python
   import time
   from live_feature_utils import build_feature_frame

   start = time.time()
   df = build_feature_frame(raw_df)  # Use sample data
   latency = (time.time() - start) * 1000
   print(f'Feature calc: {latency:.0f}ms')
   ```

**Fix:**
- Use faster timeframe (5T instead of 1H)
- Pre-calculate features and cache
- Move to faster server / closer to exchange

---

### Issue: Execution Errors / Broker Rejects

**Check:**
1. TP/SL distance meets broker minimum
   ```python
   from market_costs import get_costs
   costs = get_costs('XAUUSD')
   print(f'Min distance: {costs.min_distance_pips} pips')
   ```

2. Position size not too small
   ```bash
   # Check recent order sizes in logs
   grep "position_size" logs/live_trading_*.log | tail -10
   ```

**Fix:**
- Increase TP/SL multipliers if hitting broker minimums
- Increase account size if position sizes too small
- Verify broker API connectivity

---

## 🔄 ROLLBACK PROCEDURE

### When to Rollback:
- Critical bug discovered
- Win rate < 30% for 20+ trades
- Drawdown > 8%
- Execution errors on >50% of signals

### Rollback Steps:
```bash
# 1. STOP live engine immediately
pkill -f live_trading_engine_fixed.py

# Or use saved PID:
kill $(cat logs/live_engine.pid)

# 2. Verify stopped
ps aux | grep live_trading_engine

# 3. Close any open positions (MANUAL via broker UI)

# 4. Switch to paper mode for diagnosis
export PAPER_MODE=true

python live_trading_engine_fixed.py \
    --symbol XAUUSD \
    --timeframe 15T \
    --paper-mode \
    > logs/rollback_paper_$(date +%Y%m%d).log 2>&1 &

# 5. Investigate root cause
tail -f logs/rollback_paper_*.log

# 6. Fix bug, run paper for 1 week, redeploy
```

---

## 🎯 PERFORMANCE TARGETS

### Minimum Acceptable Performance (Live)
| Metric | Target | Red Flag |
|--------|--------|----------|
| Win Rate | ≥ 50% | < 45% |
| Profit Factor | ≥ 1.3 | < 1.1 |
| Avg R-multiple | ≥ 0.20R | < 0.10R |
| Max Drawdown | ≤ 6% | > 8% |
| Sharpe/trade | ≥ 0.20 | < 0.10 |
| Execution Latency | < 250ms | > 500ms |
| Signals/week | 10-30 | < 5 or > 50 |

### World-Class Performance (Stretch Goals)
| Metric | Target |
|--------|--------|
| Win Rate | ≥ 55% |
| Profit Factor | ≥ 1.6 |
| Avg R-multiple | ≥ 0.30R |
| Max Drawdown | ≤ 4% |
| Sharpe/trade | ≥ 0.30 |

---

## ⚙️ CONFIGURATION REFERENCE

### Environment Variables
```bash
# Required
POLYGON_API_KEY=your_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key_here

# Optional
RISK_PER_TRADE_PCT=0.01          # 1% risk per trade
PAPER_MODE=false                  # true = paper trading
MAX_DAILY_TRADES=10               # Limit trades per day
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
```

### market_costs.py Configuration
Edit `market_costs.py` to adjust:
```python
# TP/SL parameters
TP_SL_PARAMS = {
    'XAUUSD': {
        '15T': SymbolTPSL(tp_atr_mult=1.4, sl_atr_mult=1.0),  # Adjust here
    }
}

# Market costs
MARKET_COSTS = {
    'XAUUSD': SymbolCosts(
        spread_pips=3.0,         # Adjust based on broker
        commission_pct=0.00002,  # Adjust based on broker
        slippage_pct=0.00001,    # Adjust based on observed slippage
    )
}
```

### execution_guardrails.py Configuration
```python
# Use preset or create custom
guards = get_moderate_guardrails()  # or get_conservative_guardrails()

# Or customize:
guards = ExecutionGuardrails(
    max_spread_atr_ratio=0.15,     # Tighten to 0.10 for better fills
    max_data_age_seconds=300,       # Reduce to 180 for faster data
    min_confidence=0.55,            # Increase to 0.60 for higher quality
    blocked_sessions=['asia'],      # Add/remove sessions
)
```

---

## 📞 SUPPORT & ESCALATION

### Severity Levels

**P0 - Critical (Immediate):**
- Trading halted due to bug
- Drawdown > 10%
- Execution errors on ALL signals
- **Action:** Stop all trading, investigate immediately

**P1 - High (Same Day):**
- Win rate < 35% for 20+ trades
- Latency > 1000ms consistently
- Guardrails failing
- **Action:** Switch to paper mode, debug

**P2 - Medium (This Week):**
- Win rate 40-45% (below target but not critical)
- Occasional execution errors
- **Action:** Monitor, investigate if persists

**P3 - Low (Backlog):**
- Feature requests
- Performance optimizations
- Documentation updates

### Contact
- **On-Call:** (Your contact info)
- **Escalation:** (Manager contact)
- **Broker Support:** (Broker phone/email)

---

## 📚 REFERENCES

- [POSTMORTEM.md](POSTMORTEM.md) - Root cause analysis of original bugs
- [market_costs.py](market_costs.py) - Cost model documentation
- [execution_guardrails.py](execution_guardrails.py) - Guardrail logic
- [Polygon API Docs](https://polygon.io/docs/forex/getting-started)
- [Supabase Python Docs](https://supabase.com/docs/reference/python/introduction)

---

## ✅ DEPLOYMENT CHECKLIST (Quick Reference)

```
Pre-Deployment:
[ ] Environment variables set
[ ] Models validated and loaded
[ ] market_costs.py tested
[ ] execution_guardrails.py tested
[ ] Feature parity validated
[ ] Backtest with new costs shows targets met

Paper Trading (Week 1):
[ ] Deployed to paper mode
[ ] 20+ signals generated
[ ] Paper WR ≥ 50%, PF ≥ 1.2
[ ] No critical bugs
[ ] Latency < 250ms

Live Deployment (0.1% risk):
[ ] Risk set to 0.001 (0.1%)
[ ] Max daily trades = 2
[ ] Monitoring enabled
[ ] Alerts configured
[ ] 10+ successful trades completed

Scale-Up:
[ ] Week 2: 0.25% risk (after 10 successful trades)
[ ] Week 3: 0.5% risk (if metrics hold)
[ ] Week 4+: 1.0% risk (if metrics hold)

Ongoing:
[ ] Daily performance review
[ ] Weekly drift monitoring
[ ] Monthly model retraining
```

---

**Last Reviewed:** 2025-11-13
**Next Review Due:** 2025-12-13

**Status:** ✅ READY FOR DEPLOYMENT (after paper validation)
