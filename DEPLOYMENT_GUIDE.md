# 🚀 DEPLOYMENT GUIDE
## Renaissance Technologies ML Trading System

---

## QUICK START

### 1. Production-Ready Models Location
```bash
/Users/omar/Desktop/ML_Trading/models_production/
```

### 2. Ready for Deployment (6 Models)
```
✅ XAGUSD/XAGUSD_15T_PRODUCTION_READY.pkl
✅ GBPUSD/GBPUSD_15T_PRODUCTION_READY.pkl
✅ GBPUSD/GBPUSD_1H_PRODUCTION_READY.pkl
✅ AUDUSD/AUDUSD_15T_PRODUCTION_READY.pkl
✅ AUDUSD/AUDUSD_1H_PRODUCTION_READY.pkl
✅ NZDUSD/NZDUSD_15T_PRODUCTION_READY.pkl
```

---

## DEPLOYMENT CHECKLIST

### Phase 1: Paper Trading (Week 1-2)
- [ ] Load all 6 models
- [ ] Connect to paper trading account
- [ ] Set risk to 0.25% per trade (half of production)
- [ ] Monitor for 2 weeks
- [ ] Verify DD < 3%
- [ ] Verify PF > 1.3

### Phase 2: Micro Live (Week 3-4)
- [ ] Scale to 0.5% risk per trade
- [ ] Deploy 10% of capital
- [ ] Monitor daily
- [ ] Track slippage vs backtest
- [ ] Verify performance within 20% of backtest

### Phase 3: Full Production (Month 2+)
- [ ] Scale to 0.5% risk per trade
- [ ] Deploy 50% of capital
- [ ] Keep 50% in reserve
- [ ] Monitor weekly
- [ ] Review monthly

---

## RISK MANAGEMENT RULES

### Hard Limits
1. **Max Risk per Trade:** 0.5%
2. **Circuit Breaker:** 7% portfolio DD → STOP ALL TRADING
3. **Daily DD Limit:** 2% → Reduce risk by 50%
4. **Weekly DD Limit:** 4% → Review all models
5. **Monthly DD Limit:** 7% → Pause for review

### Position Sizing by Symbol
```python
POSITION_SIZES = {
    'XAGUSD': 0.30,  # 30% of standard (high volatility)
    'GBPUSD': 0.50,  # 50% of standard
    'AUDUSD': 0.70,  # 70% of standard
    'NZDUSD': 0.60,  # 60% of standard
}
```

### Signal Parameters by Model
```python
MODEL_PARAMS = {
    'XAGUSD_15T': {
        'tp': 1.5, 'sl': 1.0,
        'min_confidence': 0.38,
        'min_edge': 0.10
    },
    'GBPUSD_15T': {
        'tp': 1.6, 'sl': 1.0,
        'min_confidence': 0.35,
        'min_edge': 0.08
    },
    'GBPUSD_1H': {
        'tp': 1.6, 'sl': 1.0,
        'min_confidence': 0.35,
        'min_edge': 0.08
    },
    'AUDUSD_15T': {
        'tp': 1.5, 'sl': 1.0,
        'min_confidence': 0.38,
        'min_edge': 0.10
    },
    'AUDUSD_1H': {
        'tp': 1.6, 'sl': 1.0,
        'min_confidence': 0.38,
        'min_edge': 0.10
    },
    'NZDUSD_15T': {
        'tp': 1.5, 'sl': 1.0,
        'min_confidence': 0.38,
        'min_edge': 0.10
    },
}
```

---

## MONITORING DASHBOARD

### Daily Checks (5 minutes)
- [ ] Current DD % vs 7% limit
- [ ] Open positions count
- [ ] P&L vs expected
- [ ] Any circuit breaker triggers?

### Weekly Review (30 minutes)
- [ ] Win rate by symbol
- [ ] Profit factor by model
- [ ] Slippage analysis
- [ ] Trade count vs backtest
- [ ] Return vs S&P500

### Monthly Deep Dive (2 hours)
- [ ] Full performance report
- [ ] Correlation analysis
- [ ] Model drift detection
- [ ] Retrain if needed
- [ ] Adjust parameters if needed

---

## EXPECTED PERFORMANCE

### Conservative Estimate (Live Trading)
- **Profit Factor:** 1.3 - 1.5 (20% degradation from backtest)
- **Win Rate:** 50% - 55%
- **Max Drawdown:** 5% - 7%
- **Sharpe Ratio:** 0.35 - 0.50
- **Monthly Return:** 3% - 6%

### Portfolio Statistics
- **Total Trades:** ~170/month (all 6 models)
- **Avg Trade Duration:** 10-20 bars
- **Long/Short Balance:** 55/45 to 45/55
- **Correlation:** Low (<0.3 between symbols)

---

## RETRAINING SCHEDULE

### When to Retrain
1. **Profit factor drops < 1.2** for 2 consecutive weeks
2. **DD exceeds 10%** (after halting trading)
3. **Win rate drops < 45%** for 1 month
4. **Every 6 months** (routine maintenance)

### How to Retrain
```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

# Retrain single model
python production_final_system.py --symbol XAGUSD --tf 15T

# Retrain all
python production_final_system.py --all
```

---

## TROUBLESHOOTING

### Issue: Excessive Slippage
**Solution:** 
- Trade during liquid hours only (8am-5pm EST)
- Avoid trading during major news events
- Use limit orders instead of market orders

### Issue: Lower Win Rate Than Backtest
**Expected:** 3-5% lower in live trading  
**Action Required if:** > 10% lower
- Check for execution issues
- Verify data feed quality
- Review slippage

### Issue: Circuit Breaker Triggered
**Immediate Actions:**
1. Stop all trading
2. Review all open positions
3. Analyze what went wrong
4. Wait 24 hours minimum
5. Resume with 50% risk

### Issue: Directional Bias
**Symptoms:** Only taking longs OR only taking shorts  
**Solution:**
- Check model probabilities
- May indicate regime change
- Consider retraining with recent data

---

## PORTFOLIO ALLOCATION

### Recommended Capital Distribution
```
Total Capital: $100,000

Allocated to Trading: $50,000 (50%)
  - XAGUSD 15T:  $10,000 (20%)  - Best Sharpe
  - AUDUSD 15T:  $10,000 (20%)  - Best PF
  - NZDUSD 15T:  $10,000 (20%)  - Best Return
  - GBPUSD 15T:  $8,000  (16%)  - Solid all-round
  - AUDUSD 1H:   $6,000  (12%)  - Diversification
  - GBPUSD 1H:   $6,000  (12%)  - Low DD

Reserve: $50,000 (50%)  - Safety buffer
```

### Risk Per Model
- With 0.5% risk per trade
- Max concurrent positions: 6 (one per model)
- Max portfolio risk at any time: 3%

---

## PERFORMANCE TRACKING

### Metrics to Track
```python
{
    'daily_pnl': [],
    'cumulative_return': 0.0,
    'current_drawdown': 0.0,
    'trades_today': 0,
    'wins': 0,
    'losses': 0,
    'avg_win': 0.0,
    'avg_loss': 0.0,
    'largest_win': 0.0,
    'largest_loss': 0.0,
    'profit_factor': 0.0,
    'sharpe_ratio': 0.0,
    'sortino_ratio': 0.0,
    'calmar_ratio': 0.0,
}
```

### Alerts to Set Up
- Email/SMS if DD > 5%
- Alert if circuit breaker triggers
- Daily summary at 5pm EST
- Weekly report on Sunday
- Monthly report on 1st of month

---

## LIVE TRADING EXAMPLE

```python
import pickle
import pandas as pd
import numpy as np

# Load model
with open('models_production/XAGUSD/XAGUSD_15T_PRODUCTION_READY.pkl', 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    features = saved['features']
    params = saved['params']

# Get live data (last 1000 bars)
df = get_live_data('XAGUSD', '15T', bars=1000)

# Add features
df = add_features(df)

# Prepare features
X = df[features].fillna(0).values[-1:,:]  # Last bar

# Get prediction
probs = model.predict_proba(X)[0]
flat_prob, long_prob, short_prob = probs

# Check signal
max_prob = max(probs)
sorted_probs = sorted(probs, reverse=True)
edge = sorted_probs[0] - sorted_probs[1]

if long_prob == max_prob and long_prob >= params['min_conf'] and edge >= params['min_edge']:
    signal = 'LONG'
elif short_prob == max_prob and short_prob >= params['min_conf'] and edge >= params['min_edge']:
    signal = 'SHORT'
else:
    signal = 'FLAT'

# Calculate position size
if signal != 'FLAT':
    current_equity = get_account_balance()
    risk_pct = 0.005  # 0.5%
    position_mult = params['pos_size']
    
    atr = df['atr14'].iloc[-1]
    entry_price = df['close'].iloc[-1]
    
    if signal == 'LONG':
        sl_price = entry_price - (atr * params['sl'])
        tp_price = entry_price + (atr * params['tp'])
    else:
        sl_price = entry_price + (atr * params['sl'])
        tp_price = entry_price - (atr * params['tp'])
    
    risk_amount = current_equity * risk_pct * position_mult
    position_size = risk_amount / abs(entry_price - sl_price)
    
    # Place order
    place_order(
        symbol='XAGUSD',
        side=signal,
        quantity=position_size,
        entry=entry_price,
        sl=sl_price,
        tp=tp_price
    )
```

---

## CONTACT & SUPPORT

### System Maintenance
- Review models quarterly
- Update features as needed
- Monitor for regime changes
- Keep documentation updated

### Questions?
- Review `PRODUCTION_STATUS.md` for metrics
- Check `REALISTIC_EXPECTATIONS.md` for caveats
- See `ALL_SYMBOLS_RESULTS.md` for detailed results

---

**Remember:** Past performance does not guarantee future results. Always manage risk appropriately.

---

*Last Updated: November 4, 2025*  
*Renaissance Technologies Standards*

