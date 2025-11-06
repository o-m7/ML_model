# INTRADAY TRADING SYSTEM
## Production-Grade Multi-Strategy System for Forex/Metals

A complete, production-ready trading system with 6 strategies across 8 symbols and multiple timeframes.

## 🎯 Overview

**Symbols**: XAUUSD, XAGUSD, EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCAD  
**Timeframes**: 5m, 15m, 30m, 1h, 2h, 4h  
**Strategies**: 6 distinct strategies optimized per timeframe

### Strategy Mapping

| Timeframe | Strategy | Description |
|-----------|----------|-------------|
| 5m | S1 | Momentum Breakout + Volume Confirmation |
| 15m | S2 | Mean-Reversion to VWAP/EMA |
| 30m | S3 | Pullback-to-Trend (Continuation) |
| 1h | S4 | Breakout + Retest |
| 2h | S5 | Momentum with ADX+ATR Regime Filter |
| 4h | S6 | Multi-Timeframe Trend Alignment |

## 📦 Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost pyyaml

# Install package
cd /Users/omar/Desktop/ML_Trading
pip install -e intraday_system/
```

## 📊 Data Requirements

Your data must be in Parquet format with the following structure:

```
feature_store/
├── XAUUSD/
│   ├── XAUUSD_5T.parquet
│   ├── XAUUSD_15T.parquet
│   ├── XAUUSD_30T.parquet
│   ├── XAUUSD_1H.parquet
│   ├── XAUUSD_2H.parquet
│   └── XAUUSD_4H.parquet
├── EURUSD/
│   └── ...
└── ...
```

### Required Columns
- `timestamp` (datetime, UTC)
- `open`, `high`, `low`, `close`, `volume` (float)

The system will automatically compute 180+ technical indicators.

## 🚀 Quick Start

### 1. Train Single Symbol/Timeframe

```bash
python -m intraday_system.cli.train \
    --symbol XAUUSD \
    --timeframe 15T \
    --data-root feature_store \
    --out models_intraday
```

### 2. Train All Symbols (Parallel)

```bash
python -m intraday_system.cli.train \
    --symbols ALL \
    --timeframes 5T,15T,30T,1H,2H,4H \
    --data-root feature_store \
    --out models_intraday \
    --workers 4
```

### 3. Run OOS Backtest

```bash
python -m intraday_system.cli.backtest \
    --models models_intraday \
    --data-root feature_store \
    --oos-months 6 \
    --report reports/
```

### 4. Export for Production

```bash
python -m intraday_system.cli.export \
    --models models_intraday \
    --out production_artifacts/
```

## 📈 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│ Raw OHLCV  →  Feature Engineering  →  Label Generation     │
│   (Parquet)      (180+ indicators)      (Triple-Barrier)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  STRATEGY LAYER                              │
├─────────────────────────────────────────────────────────────┤
│ S1-S6: Deterministic rule features + regime detection       │
│ • Breakout strength  • Mean reversion  • Trend alignment    │
│ • Volume confirmation  • Momentum filters                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    ML LAYER                                  │
├─────────────────────────────────────────────────────────────┤
│ Ensemble: LightGBM (40%) + XGBoost (40%) + Linear (20%)    │
│ • Walk-Forward CV (10 folds, purged + embargoed)           │
│ • Class balancing  • Feature selection  • Regularization    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION & SELECTION                       │
├─────────────────────────────────────────────────────────────┤
│ Go-Live Benchmarks:                                          │
│ • Profit Factor ≥ 1.60     • Max DD ≤ 6%                   │
│ • Sharpe/trade ≥ 0.25      • Win Rate ≥ 52%                │
│ • Min Trades ≥ 200         • Stress tests pass              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   LIVE API                                   │
├─────────────────────────────────────────────────────────────┤
│ predict(symbol, timeframe, latest_bars) → Signal            │
│ • BUY/SELL/HOLD  • Confidence  • SL/TP  • Horizon          │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 Feature Engineering

The system automatically computes 180+ features:

### Base Indicators
- **Trend**: EMA(10,20,50,100,200), SMA(10,20,50,100,200)
- **Momentum**: RSI(14), MACD(12,26,9), ADX(14)
- **Volatility**: ATR(14,20), Bollinger Bands(20,2), Parkinson Vol
- **Support/Resistance**: Donchian(20), VWAP(100)
- **Volume**: Volume MA, OBV, Volume spikes

### Strategy-Specific Features
Each strategy adds 15-20 specialized features (see strategies/*.py)

### Regime Features
- Trend strength (ADX-based)
- Volatility regimes (ATR percentiles)
- EMA slope alignment
- Ranging vs trending detection

## 📋 Go-Live Benchmarks

Models must pass ALL criteria to be marked production-ready:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Profit Factor | ≥ 1.60 | Wins/losses ratio |
| Max Drawdown | ≤ 6.0% | Risk control |
| Sharpe/Trade | ≥ 0.25 | Risk-adjusted returns |
| Win Rate | ≥ 52% | Edge confirmation |
| Min Trades (OOS) | ≥ 200 | Statistical significance |

### Stress Tests
Models are tested with:
- +25% trading costs
- ±1 bar latency
- Slippage jitter
- Feature noise

**Stressed Thresholds**: PF ≥ 1.30, DD ≤ 8%

## 🔄 Walk-Forward Cross-Validation

```
2019 ────────────────────────────────────── 2025
│
├──── Fold 1 ────┤  Train  │ Val │←─ Embargo
├──────── Fold 2 ────────┤  Train  │ Val │
├────────── Fold 3 ──────────┤  Train  │ Val │
...
├──────────────── Fold 10 ──────────────┤ Train │ Val │
│                                                  │
└──────────────────────────────── OOS (6mo) ──────┘
```

- **10 folds** expanding window
- **Purge**: 50 bars between train/val
- **Embargo**: 100 bars after validation
- **OOS**: Final 6 months held out

## 🎯 Live Inference API

```python
from intraday_system.live import predict

# Get latest bars for a symbol
latest_data = load_latest_bars("XAUUSD", "15T", n_bars=200)

# Generate signal
signal = predict(
    symbol="XAUUSD",
    timeframe="15T",
    latest_bars=latest_data
)

print(signal)
# {
#     'signal': 'BUY',
#     'confidence': 0.73,
#     'entry_ref': 2651.50,
#     'stop_loss': 2648.30,
#     'take_profit': 2656.10,
#     'expected_R': 1.44,
#     'horizon_bars': 8,
#     'timestamp_utc': '2025-11-03T14:30:00Z',
#     'expiry_bar_index': 12345678
# }
```

## 📁 Output Artifacts

After training, each symbol/timeframe produces:

```
models_intraday/
└── XAUUSD/
    └── 15T/
        ├── model.pkl                 # Trained ensemble
        ├── scaler.pkl                # Feature scaler
        ├── features.json             # Feature list
        ├── model_card.json           # Metadata & metrics
        └── performance_report.html   # Visual report
```

### Model Card Example
```json
{
  "symbol": "XAUUSD",
  "timeframe": "15T",
  "strategy": "S2",
  "status": "READY",
  "training_window": "2019-01-01 to 2025-04-22",
  "n_features": 47,
  "cv_metrics": {
    "mean_accuracy": 0.58,
    "std_accuracy": 0.03
  },
  "oos_metrics": {
    "profit_factor": 1.82,
    "sharpe_per_trade": 0.31,
    "win_rate": 54.2,
    "max_drawdown_pct": 4.8,
    "total_trades": 287
  },
  "benchmarks_passed": true
}
```

## 🧪 Testing

```bash
# Run all tests
pytest intraday_system/tests/

# Specific test suites
pytest intraday_system/tests/test_leakage.py
pytest intraday_system/tests/test_labels.py
pytest intraday_system/tests/test_features.py
pytest intraday_system/tests/test_eval.py
```

## ⚙️ Configuration

### Main Settings (`config/settings.yaml`)
- Data paths and date ranges
- Trading costs (commission, slippage, spreads)
- Risk parameters (position sizing, stops)
- Walk-forward CV settings
- Go-live benchmarks

### Strategy Params (`config/strategies.yaml`)
- Per-strategy hyperparameters
- Label horizons and R multiples
- Confidence thresholds
- Entry/exit rules

## 🔍 Debugging Failed Models

If a model fails benchmarks:

```python
# Load model card
import json
with open('models_intraday/XAUUSD/15T/model_card.json') as f:
    card = json.load(f)

print(card['oos_metrics'])
print(card['failures'])  # Shows which benchmarks failed

# Common issues:
# 1. Insufficient trades → Lower confidence threshold
# 2. Low win rate → Check strategy logic, increase TP/SL ratio
# 3. High drawdown → Tighter stops, better regime filtering
# 4. Poor PF → Feature engineering, longer horizons
```

## 📊 Example Results

Typical performance for passing models:

| Symbol | TF | PF | Sharpe | WR% | DD% | Trades |
|--------|----|----|--------|-----|-----|--------|
| EURUSD | 15T | 1.72 | 0.28 | 53.1 | 5.2 | 312 |
| XAUUSD | 5T | 1.85 | 0.33 | 55.8 | 4.7 | 428 |
| GBPUSD | 1H | 1.64 | 0.26 | 52.4 | 5.9 | 218 |

Not all symbol/TF combinations will pass - this is expected and correct behavior.

## 🚨 Production Deployment Notes

1. **Cooldown**: System enforces 5-bar cooldown between trades
2. **Spread Filter**: Skips signals if spread > 0.5 × ATR
3. **Expiry**: Signals expire after horizon_bars
4. **Position Sizing**: 1% risk per trade (configurable)
5. **Max Position**: 10% of capital per trade

## 📚 Further Reading

- `features/builders.py` - Technical indicator computations
- `labels/triple_barrier.py` - Labeling methodology
- `strategies/` - Individual strategy implementations
- `models/ensembles.py` - Ensemble architecture
- `evaluation/` - Performance measurement

## 🤝 Support

For issues or questions:
1. Check model cards for detailed failure reasons
2. Review strategy-specific features
3. Adjust config parameters
4. Run stress tests to identify weaknesses

## 📝 License

Production trading system - Use at your own risk. Past performance does not guarantee future results.

---

**Built with**: Python 3.10+, scikit-learn, LightGBM, XGBoost, pandas, numpy

