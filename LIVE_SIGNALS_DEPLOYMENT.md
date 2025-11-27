# Live Signal Generator - Production Deployment Guide

## 🎯 Overview

A production-ready real-time trading signal generator that:
- Streams live market data from Polygon.io via WebSocket
- Computes trading features on closed bars only (no look-ahead bias)
- Generates consensus signals using ONNX models
- Persists signals to Supabase for frontend consumption
- Deploys on GitHub Actions with automatic market hours scheduling

**Technology Stack**:
- Python 3.10+
- Polygon.io (real-time market data)
- XGBoost + ONNX (ML models)
- Supabase (signal storage + realtime)
- GitHub Actions (CI/CD & continuous deployment)

---

## 📋 Prerequisites

### 1. Environment Variables (.env file)

```env
# Polygon.io
POLYGON_API_KEY=jVLDXLylHzIpygLbXc0oYuuMGKnNOqpx

# Supabase
SUPABASE_URL=https://ifetofkhyblyijghuwzs.supabase.co
SUPABASE_KEY=sb_secret_qqxFcwWf3CHKL7kDNoMdng_ezMmtbv8
```

### 2. Supabase Setup

Create a table for signals (run in Supabase SQL editor):

```sql
CREATE TABLE signals (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    signal INTEGER NOT NULL CHECK (signal IN (-1, 0, 1)),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_output JSONB,
    features JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX idx_signals_symbol_timeframe ON signals(symbol, timeframe);
CREATE INDEX idx_signals_created_at ON signals(created_at DESC);

-- Enable realtime
ALTER PUBLICATION supabase_realtime ADD TABLE signals;
```

### 3. GitHub Secrets

Add these to your GitHub repository settings:

```
POLYGON_API_KEY: <your_polygon_api_key>
SUPABASE_URL: <your_supabase_url>
SUPABASE_KEY: <your_supabase_key>
```

---

## 🚀 Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements_live_signals.txt
```

2. **Create .env file** with credentials

3. **Convert models to ONNX** (one-time):
```bash
python model_converter.py
```

4. **Run test mode** (generates signals from synthetic data):
```bash
python run_live_signals.py --test
```

5. **Run live mode** (connects to Polygon.io):
```bash
python run_live_signals.py
```

### Docker Deployment

```bash
# Build image
docker build -t live-signals .

# Run container
docker run \
  -e POLYGON_API_KEY=<key> \
  -e SUPABASE_URL=<url> \
  -e SUPABASE_KEY=<key> \
  live-signals
```

---

## 🔄 Architecture

### Data Flow

```
Polygon.io WebSocket
      ↓
AggBar (minute candles)
      ↓
BarBuilder (aggregate into 5T, 15T, 30T)
      ↓
FeatureEngine (compute 17+ features)
      ↓
ONNXModelInterface (ensemble voting)
      ↓
Signal (symbol, timeframe, signal, confidence)
      ↓
Supabase (persistent storage)
      ↓
Frontend (realtime display)
```

### Component Breakdown

| Component | Purpose |
|-----------|---------|
| `live_signal_engine.py` | Core signal generation logic |
| `polygon_connector.py` | WebSocket client for Polygon data |
| `supabase_store.py` | Signal persistence layer |
| `model_converter.py` | XGBoost → ONNX conversion |
| `run_live_signals.py` | Main orchestrator + CLI |

---

## 📊 Feature Engineering

### Computed Features (17 total)

```python
# Price & Returns
- c: close price
- ret: log returns
- vol_20: 20-bar volatility
- ret_zscore_20: normalized returns

# Volatility & ATR
- atr: average true range
- atr_pct: ATR as % of price

# Momentum
- rsi_14: relative strength index
- ma_diff: fast EMA - slow EMA
- ma_ratio: fast EMA / slow EMA

# Trend Flags
- trend_up: 1 if uptrend
- trend_down: 1 if downtrend

# Volume
- v: volume
- vol_zscore_20: normalized volume
- vol_spike: 1 if volume spike detected
- vol_ratio: volume / volume MA

# Spreads & Quotes
- spread_last: bid-ask spread
- spread_pct_last: spread as %
- mid_last: bid-ask midpoint
```

### No Look-Ahead Guarantee

- Features computed from **closed bars only**
- No `.shift(-1)` or future data access
- Features use rolling windows (past + current)
- Model input is always at bar end timestamp

---

## 🎛️ Configuration

Edit `live_signal_engine.py` `Config` class to customize:

```python
class Config:
    # Symbol & timeframes
    SYMBOL = "XAUUSD"  # Change to other Polygon tickers
    TIMEFRAMES = {
        "1T": 60,      # 1 minute
        "5T": 300,     # 5 minutes
        "15T": 900,    # 15 minutes
        "30T": 1800,   # 30 minutes
    }
    
    # Feature windows
    RSI_WINDOW = 14
    ATR_WINDOW = 14
    MA_FAST = 10
    MA_SLOW = 50
    VOL_WINDOW = 20
    
    # Thresholds
    MIN_BARS_FOR_FEATURES = 50  # Warmup period
    SIGNAL_CONFIDENCE_THRESHOLD = 0.5  # Min confidence to emit
```

---

## 📤 Signal Output Format

Each signal is a JSON record stored in Supabase:

```json
{
  "symbol": "XAUUSD",
  "timeframe": "5T",
  "timestamp": "2024-11-26T15:30:00+00:00",
  "signal": 1,
  "confidence": 0.87,
  "model_output": {
    "ohlcv": {
      "signal": 1,
      "confidence": 0.89,
      "probabilities": [0.05, 0.06, 0.89]
    },
    "quote": {
      "signal": 1,
      "confidence": 0.85,
      "probabilities": [0.08, 0.07, 0.85]
    },
    "ensemble_avg": 0.87
  },
  "features": {
    "c": 2050.5,
    "ret": 0.0015,
    "rsi_14": 65.3,
    ...
  }
}
```

---

## 🔌 Polygon.io API

### Supported Symbols

```
Forex Pairs (FX):
- AUDUSD (Australian Dollar)
- EURUSD (Euro)
- GBPUSD (British Pound)
- XAUUSD (Gold) ← Default
- XAGUSD (Silver)
- XPTUSD (Platinum)
- XPDUSD (Palladium)

Crypto:
- BTC/USD
- ETH/USD
- etc.

Indices:
- SPX (S&P 500)
- DXY (Dollar Index)
```

### API Rate Limits

- Free tier: 5 requests/minute
- WebSocket: Unlimited during subscription
- Recommended: Use WebSocket for continuous data

---

## 🏃 GitHub Actions - Continuous Deployment

The workflow runs signal generation on GitHub Actions infrastructure:

### Schedule

```yaml
# Market hours: Monday-Friday, every 5 minutes
- cron: '*/5 * * * 1-5'
```

### Jobs

1. **test-signals** (daily)
   - Runs on all branches
   - Tests signal generation with synthetic data
   - Verifies ONNX models
   
2. **live-stream** (on push)
   - Continuous signal generation (6 hour sessions)
   - Stores results to Supabase
   - Runs whenever code is pushed

### Logs

View workflow logs in GitHub Actions tab:
```
Repo → Actions → Live Signal Generator
```

---

## 🧪 Testing

### Test Mode (No Polygon Connection)

```bash
python run_live_signals.py --test
```

Generates 100 synthetic bars and tests end-to-end pipeline:
- Feature computation
- Model inference
- Signal generation

### Unit Tests

```bash
pytest tests/test_signal_engine.py -v
pytest tests/test_features.py -v
pytest tests/test_bars.py -v
```

### Manual Testing

```python
from live_signal_engine import SignalGenerator, AggBar
import pandas as pd

gen = SignalGenerator()

# Create test bar
bar = AggBar(
    symbol="XAUUSD",
    o=2000, h=2002, l=1998, c=2001, v=1000,
    start_ts=pd.Timestamp('2024-01-01', tz='UTC'),
    end_ts=pd.Timestamp('2024-01-01 00:01', tz='UTC'),
)

# Generate signals
signals = gen.process_polygon_agg(bar)
print(signals)
```

---

## 🐛 Troubleshooting

### Issue: "POLYGON_API_KEY not in environment"

**Solution**: 
```bash
# Check .env file exists
ls -la .env

# Or set directly
export POLYGON_API_KEY=your_key
```

### Issue: "No module named 'websockets'"

**Solution**:
```bash
pip install websockets
```

### Issue: "ONNX models not found"

**Solution**:
```bash
# Run conversion script
python model_converter.py

# Verify output
ls artifacts/onnx_models/
```

### Issue: "Supabase connection refused"

**Solution**:
```bash
# Check credentials in .env
# Verify Supabase project is active
# Check network connectivity
curl https://your_supabase_url/rest/v1/signals -H "apikey: your_key"
```

---

## 📈 Performance Monitoring

### Key Metrics

Track in logs:
```
- Bars processed per second
- Signals generated per session
- Model inference latency
- Feature computation time
- Supabase write success rate
```

### Example Output

```
2024-11-26 15:30:45 | live_signal_engine | INFO | 
📊 SIGNAL: XAUUSD | 5T | Direction: +1 | Confidence: 87%

2024-11-26 15:31:00 | supabase_store | INFO | 
✓ Stored signal to Supabase: XAUUSD 5T
```

---

## 🔐 Security

### Best Practices

1. **Never commit .env file**:
```bash
# In .gitignore
.env
.env.local
credentials/
```

2. **Use GitHub Secrets** for API keys (not environment variables in code)

3. **Rotate API keys regularly**:
   - Polygon.io: Generate new key, update GitHub secret
   - Supabase: Use service role key for backend only

4. **Restrict Supabase access**:
   - Use Row Level Security (RLS) policies
   - Limit realtime subscriptions by user

---

## 📚 API Reference

### SignalGenerator

```python
from live_signal_engine import SignalGenerator

gen = SignalGenerator()

# Process aggregate
signals = gen.process_polygon_agg(agg_bar)

# Process quote
gen.process_quote(quote)

# Get latest signal
latest = gen.get_latest_signal("5T")

# Export signals
json_str = gen.get_signals_json()
```

### SupabaseSignalStore

```python
from supabase_store import SupabaseSignalStore

store = SupabaseSignalStore()

# Store signal
store.store_signal(signal)

# Retrieve signals
signals = store.get_latest_signals(symbol="XAUUSD", limit=100)

# Get by timeframe
signals = store.get_signals_by_timeframe("XAUUSD", "5T", limit=50)
```

### ModelConverter

```python
from model_converter import ModelConverter

converter = ModelConverter()

# Convert all models
converter.convert_all_models()

# Verify ONNX
converter.verify_onnx_model("artifacts/onnx_models/model.onnx")

# Test inference
converter.test_onnx_inference("artifacts/onnx_models/model.onnx")
```

---

## 🔄 CI/CD Pipeline

```
Push to main
    ↓
GitHub Actions triggers
    ↓
Install dependencies
    ↓
Convert models to ONNX
    ↓
Run test mode
    ↓
(On success) Start live stream
    ↓
Generate signals continuously
    ↓
Store to Supabase
    ↓
Frontend updates realtime
```

---

## 📞 Support & Troubleshooting

### Logs Location

**Local**:
```bash
tail -f logs/signals.log
```

**GitHub Actions**:
```
Repo → Actions → Live Signal Generator → Workflow run → signal-stream
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| WebSocket timeout | Check Polygon API status, increase timeout |
| ONNX inference slow | Verify GPU available, check batch size |
| Supabase timeouts | Check network, reduce batch size, add retry logic |
| Memory bloat | Implement rolling window cleanup (done in code) |

---

## 🎓 Next Steps

1. **Monitor signals** in Supabase realtime
2. **Build frontend** that subscribes to signal updates
3. **Add risk management** (position sizing, stop-loss)
4. **Integrate with broker** (execute trades based on signals)
5. **Track P&L** and model performance

---

## 📄 License

Proprietary - Omar Trading System

---

**Last Updated**: November 26, 2024
**Version**: 1.0.0-live
