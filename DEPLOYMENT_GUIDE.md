# Live Signals Deployment Guide

## ✅ System Status

**Production-ready signal generator with model-adaptive TP/SL completed!**

- 8 trained models (OHLCV + Quote for 1T, 5T, 15T, 30T)
- Model-adaptive TP/SL (confidence-based R:R 1.9-2.97)
- REST API optimized (<200ms latency)
- Complete trading parameters (Entry, TP, SL, Order Type)
- Table formatting and Supabase integration ready

## 📦 What's Deployed to GitHub

Successfully pushed to: `claude/fix-trading-model-performance-01HMR2zbUJUWu8GAmRhbf3vK`

**Core Files:**
- `live_signals_production.py` - Main production signal generator
- `polygon_connector.py` - Fast REST API client (11-13ms quotes)
- `train_all_ohlcv_models.py` & `train_all_quote_models.py` - Model training scripts
- `artifacts/` - 8 trained models (1T, 5T, 15T, 30T x OHLCV/Quote)
- `create_supabase_schema.sql` - Database table schema
- `.github/workflows/live-signals.yml` - GitHub Actions workflow (needs update)

## 🚀 Quick Start (Local Testing)

```bash
# 1. Set environment variables
export POLYGON_API_KEY="jVLDXLylHzIpygLbXc0oYuuMGKnNOqpx"
export SUPABASE_URL="https://ifetofkhyblyijghuwzs.supabase.co"
export SUPABASE_KEY="your_supabase_key_here"

# 2. Install dependencies
pip install aiohttp xgboost pandas numpy tabulate supabase

# 3. Run signal generator (once)
python -c "
import asyncio
from live_signals_production import TradingSignalGenerator

async def test():
    generator = TradingSignalGenerator(api_key='jVLDXLylHzIpygLbXc0oYuuMGKnNOqpx')
    generator.load_models()
    signals = await generator.generate_all_signals()
    print(generator.format_signals_table(signals))

asyncio.run(test())
"
```

**Expected Output:**
```
TF   Model       Signal       Conf    Order        Entry (Mkt)  Entry (Lmt)  TP        SL        R:R
30T  quote_30T   SELL_MARKET  84.4%   SELL_MARKET  $4151.46     $4151.77     $4146.40  $4153.16  2.97
1T   ohlcv_1T    SELL_LIMIT   54.3%   SELL_LIMIT   $4151.46     $4151.77     $4149.65  $4152.41  1.9
```

## ☁️ Deployment Options

### Option 1: GitHub Actions (Scheduled)

**Status:** Workflow file exists but needs update

**Current Issue:** `.github/workflows/live-signals.yml` references old architecture (WebSocket streaming)

**To Fix:**
1. Update workflow to run `live_signals_production.py` instead
2. Add secrets to GitHub repository:
   - `POLYGON_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`

**Modified workflow:**
```yaml
name: Live Signal Generator

on:
  schedule:
    - cron: '*/5 * * * *'  # Every 5 minutes

jobs:
  generate-signals:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - run: pip install aiohttp xgboost pandas numpy tabulate supabase
    - name: Generate signals
      run: |
        python -c "
        import asyncio
        from live_signals_production import TradingSignalGenerator
        
        async def main():
            gen = TradingSignalGenerator(api_key='${{ secrets.POLYGON_API_KEY }}')
            gen.load_models()
            signals = await gen.generate_all_signals()
            if signals:
                print(gen.format_signals_table(signals))
                gen.send_to_supabase(signals, '${{ secrets.SUPABASE_URL }}', '${{ secrets.SUPABASE_KEY }}')
        
        asyncio.run(main())
        "
```

### Option 2: Railway (Continuous)

**Best for:** Continuous operation with logs

```bash
# 1. Install Railway CLI
curl -fsSL https://railway.app/install.sh | sh

# 2. Login and create project
railway login
railway init

# 3. Set environment variables
railway variables set POLYGON_API_KEY=jVLDXLylHzIpygLbXc0oYuuMGKnNOqpx
railway variables set SUPABASE_URL=https://ifetofkhyblyijghuwzs.supabase.co
railway variables set SUPABASE_KEY=your_key_here

# 4. Create requirements.txt
cat > requirements_live.txt << EOF
aiohttp>=3.9.0
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
tabulate>=0.9.0
supabase>=2.0.0
EOF

# 5. Create Procfile for continuous running
cat > Procfile << EOF
worker: while true; do python -c "import asyncio; from live_signals_production import TradingSignalGenerator; gen = TradingSignalGenerator(api_key=os.getenv('POLYGON_API_KEY')); asyncio.run(gen.generate_all_signals())"; sleep 300; done
EOF

# 6. Deploy
railway up
```

### Option 3: Docker (Self-Hosted)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy code
COPY live_signals_production.py polygon_connector.py ./
COPY artifacts/ ./artifacts/

# Install dependencies
RUN pip install aiohttp xgboost pandas numpy tabulate supabase

# Run continuously
CMD while true; do \
    python -c "import asyncio; from live_signals_production import TradingSignalGenerator; gen = TradingSignalGenerator(api_key='${POLYGON_API_KEY}'); gen.load_models(); asyncio.run(gen.generate_all_signals())"; \
    sleep 300; \
done
```

**Run:**
```bash
docker build -t live-signals .
docker run -e POLYGON_API_KEY=xxx -e SUPABASE_URL=xxx -e SUPABASE_KEY=xxx live-signals
```

## 🗄️ Supabase Setup

**1. Create Table**

Go to: https://app.supabase.com/project/ifetofkhyblyijghuwzs/sql

Run SQL from `create_supabase_schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS trading_signals (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model_name TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    entry_market DOUBLE PRECISION NOT NULL,
    entry_limit DOUBLE PRECISION NOT NULL,
    take_profit DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION NOT NULL,
    order_type TEXT NOT NULL,
    atr DOUBLE PRECISION,
    spread DOUBLE PRECISION,
    risk DOUBLE PRECISION,
    reward DOUBLE PRECISION,
    rr_ratio DOUBLE PRECISION,
    current_bid DOUBLE PRECISION,
    current_ask DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_timeframe ON trading_signals(timeframe);
CREATE INDEX IF NOT EXISTS idx_signals_model ON trading_signals(model_name);
```

**2. Get API Keys**

Settings → API → Copy:
- Project URL: `https://ifetofkhyblyijghuwzs.supabase.co`
- anon/public key: Use for read access
- service_role key: Use for write access (keep secret!)

**3. Verify Table**

```sql
SELECT * FROM trading_signals ORDER BY created_at DESC LIMIT 10;
```

## 📊 Model Performance

**OHLCV Models (trained Nov 26, 2025):**
- 1T: 49.4% accuracy
- 5T: 51.4% accuracy
- 15T: 51.0% accuracy
- 30T: 52.4% accuracy

**Quote Models (trained Nov 26, 2025):**
- 1T: 92.5% accuracy, 0.99 AUC ⭐
- 5T: 66.4% accuracy
- 15T: 66.3% accuracy
- 30T: 64.0% accuracy

**Training Data:**
- Location: `feature_store/C:XAU-USD/`
- Created: Nov 26, 2025 01:32-14:03
- Symbol: XAU/USD (Gold spot price)

## 🎯 Model-Adaptive TP/SL

The system intelligently adjusts Take Profit and Stop Loss based on model confidence:

**High Confidence (≥75%):**
- TP: 2.0-3.0x ATR (aggressive profit-taking)
- SL: 0.8x ATR (tight stops)
- Example: 30T Quote at 84.4% confidence → R:R 2.97

**Medium Confidence (55-75%):**
- TP: 2.0x ATR (balanced)
- SL: 1.0x ATR (standard)
- Example: 5T OHLCV at 66.2% confidence → R:R 2.0

**Low Confidence (<55%):**
- TP: 1.4x ATR (conservative targets)
- SL: 1.2x ATR (wider stops for safety)
- Example: 1T OHLCV at 54.3% confidence → R:R 1.9

**Additional Features:**
- Support/Resistance level detection (when 50+ bars available)
- Volatility adjustment (high vol → wider SL)
- Spread integration

See `MODEL_ADAPTIVE_TPSL.md` for full details.

## 🔧 Configuration

**API Endpoints:**
- Quotes: `https://api.polygon.io/v2/last/nbbo/C:XAU-USD`
- Bars: `https://api.polygon.io/v2/aggs/ticker/C:XAUUSD/range/{multiplier}/{timespan}/{from}/{to}`

**Symbol Formats:**
- REST Quotes: `C:XAU-USD` (with hyphen)
- REST Bars: `C:XAUUSD` (no hyphen)
- WebSocket: `XAU/USD` (with slash)

**Timeframes:**
- 1T: 1 minute
- 5T: 5 minutes
- 15T: 15 minutes
- 30T: 30 minutes

**Performance:**
- Quote fetch: 11-13ms
- Bars fetch: 32-195ms
- Signal generation: <400ms total

## 📝 Next Steps

1. **Create Supabase Table** (5 min)
   - Run SQL from `create_supabase_schema.sql`
   - Get service_role API key

2. **Configure GitHub Secrets** (5 min)
   - Go to: https://github.com/o-m7/ML_model/settings/secrets/actions
   - Add: POLYGON_API_KEY, SUPABASE_URL, SUPABASE_KEY

3. **Update GitHub Actions Workflow** (10 min)
   - Modify `.github/workflows/live-signals.yml`
   - Use simplified Python command from Option 1 above

4. **Test Deployment** (5 min)
   - Push changes to trigger workflow
   - Check Actions tab for logs
   - Verify signals in Supabase

5. **Monitor** (ongoing)
   - Check Supabase for new signals every 5 minutes
   - Review signal quality and confidence distribution
   - Adjust threshold or retrain models as needed

## 🆘 Troubleshooting

**Issue: No signals generated**
- Check API key is valid
- Verify models loaded (8 total expected)
- Ensure market is open (Mon-Fri)

**Issue: Supabase connection failed**
- Verify table exists (run SQL schema)
- Check API keys are correct
- Use service_role key for writes (not anon key)

**Issue: Models not found**
- Ensure `artifacts/` directory is deployed
- Check file paths: `artifacts/ohlcv_model_{1T,5T,15T,30T}_xgb.pkl`
- Verify git didn't exclude PKL files

**Issue: GitHub Actions timeout**
- Reduce frequency (every 15 min instead of 5)
- Use Railway instead for continuous operation
- Check quota limits

## 📚 Additional Documentation

- `MODEL_ADAPTIVE_TPSL.md` - Full explanation of adaptive TP/SL system
- `SIGNALS_SYSTEM_README.md` - Complete system documentation
- `create_supabase_schema.sql` - Database schema
- `polygon_connector.py` - REST API client implementation

## 🎉 Success Metrics

**System is working when:**
- ✅ 6-8 signals generated per run
- ✅ Confidence scores range 54%-92%
- ✅ R:R ratios range 1.9-2.97
- ✅ API latency <400ms
- ✅ Signals inserted to Supabase
- ✅ Table formatted correctly

**Example successful output:**
```
Generated 6 signals successfully

TF   Model       Signal       Conf    Order        Entry (Mkt)  Entry (Lmt)  TP        SL        R:R
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
30T  quote_30T   SELL_MARKET  84.4%   SELL_MARKET  $4151.46     $4151.77     $4146.40  $4153.16  2.97
5T   ohlcv_5T    BUY_LIMIT    66.2%   BUY_LIMIT    $4151.46     $4150.84     $4153.84  $4150.29  2.0
15T  quote_15T   SELL_LIMIT   66.3%   SELL_LIMIT   $4151.46     $4151.77     $4148.96  $4152.67  2.01
1T   ohlcv_1T    SELL_LIMIT   54.3%   SELL_LIMIT   $4151.46     $4151.77     $4149.65  $4152.41  1.9
```
