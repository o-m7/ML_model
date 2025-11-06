# 🚀 Quick Start: Auto-Upload Features to Supabase

## ⚡ 3-Step Setup (5 minutes)

### Step 1: Create Supabase Table (2 minutes)

1. **Open Supabase SQL Editor:**
   - Go to: https://ifetofkhyblyijghuwzs.supabase.co
   - Click **SQL Editor** in the left sidebar

2. **Copy & Paste the SQL:**
   - Open the file: `supabase_schema_features.sql`
   - Or run: `cat supabase_schema_features.sql`
   - Copy ALL the SQL code

3. **Execute:**
   - Paste into SQL Editor
   - Click **Run** (or press Cmd/Ctrl + Enter)
   - ✅ You should see: "Success. No rows returned"

---

### Step 2: Test Upload (1 minute)

Upload features for XAUUSD once to test:

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

python upload_features_to_supabase.py --symbol XAUUSD --timeframe 1H
```

Expected output:
```
✓ FeatureUploader initialized
Processing XAUUSD 1H
Fetching 200 bars for XAUUSD 1H...
✓ Fetched 200 bars
Calculating features for XAUUSD 1H...
✓ Calculated 264 features
✓ Uploaded features for XAUUSD 1H at 2025-11-03...
✓ Complete: 1/1 successful
```

---

### Step 3: Run Continuous Upload (2 minutes)

Start automatic feature updates every 15 minutes:

```bash
python upload_features_to_supabase.py \
    --symbols XAUUSD,EURUSD \
    --timeframes 1H,4H \
    --continuous \
    --interval 900
```

This will:
- ✅ Update features for XAUUSD and EURUSD
- ✅ Process both 1H and 4H timeframes  
- ✅ Refresh every 15 minutes (900 seconds)
- ✅ Log everything to `feature_uploader.log`
- ✅ Run until you stop it (Ctrl+C)

---

## 📊 Verify It's Working

### Check the Logs

In another terminal:
```bash
tail -f /Users/omar/Desktop/ML_Trading/feature_uploader.log
```

### Query from Supabase

Go to SQL Editor and run:
```sql
-- See latest features
SELECT 
    symbol,
    timeframe,
    timestamp,
    close,
    feature_count,
    features->>'rsi_14' as rsi,
    features->>'macd' as macd
FROM public.features
ORDER BY timestamp DESC
LIMIT 10;

-- Count records
SELECT COUNT(*) FROM public.features;
```

### Query from Python

```python
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

# Get latest XAUUSD data
result = supabase.table('features')\
    .select('*')\
    .eq('symbol', 'XAUUSD')\
    .eq('timeframe', '1H')\
    .order('timestamp', desc=True)\
    .limit(1)\
    .execute()

print(result.data[0])
```

---

## 🔄 Run in Background

### Option 1: Using screen (Recommended for testing)

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

# Start in background
screen -dmS features python upload_features_to_supabase.py \
    --symbols XAUUSD,EURUSD \
    --timeframes 1H,4H \
    --continuous

# Check it's running
screen -ls

# View output
screen -r features

# Detach: Ctrl+A then D
# Stop: screen -r features, then Ctrl+C
```

### Option 2: Using nohup (Simple background)

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

nohup python upload_features_to_supabase.py \
    --symbols XAUUSD,EURUSD \
    --timeframes 1H,4H \
    --continuous &

# Check if running
ps aux | grep upload_features

# Stop (get PID from ps command above)
kill <PID>
```

---

## 📋 What Gets Uploaded

For each symbol/timeframe, the system uploads:

### OHLCV Data
- Open, High, Low, Close, Volume
- Timestamp

### 264 Calculated Features
- **TA-Lib Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ADX, Stochastic, etc. (180+)
- **Custom Features**: Regime detection, volatility percentile, trend strength
- **Pattern Recognition**: 50+ candlestick patterns
- **Volume Indicators**: OBV, AD, ADOSC
- **Volatility**: ATR, NATR, Bollinger Bandwidth
- **Statistical**: Linear regression, correlation, standard deviation

### Storage Format
```json
{
  "id": 123,
  "symbol": "XAUUSD",
  "timeframe": "1H",
  "timestamp": "2025-11-03T22:00:00+00:00",
  "open": 4000.50,
  "high": 4005.25,
  "low": 3998.75,
  "close": 4001.18,
  "volume": 12500,
  "features": {
    "rsi_14": 55.2,
    "macd": 1.23,
    "bb_upper_20": 4010.5,
    "atr": 14.28,
    "ema_20": 4000.3,
    ...264 total features
  },
  "feature_count": 264,
  "data_source": "Polygon API",
  "created_at": "2025-11-03T22:35:00+00:00"
}
```

---

## 🎯 Use Cases

### Real-Time Signal Generation
```python
# Fetch latest features from Supabase
features = supabase.table('features')\
    .select('features')\
    .eq('symbol', 'XAUUSD')\
    .eq('timeframe', '1H')\
    .order('timestamp', desc=True)\
    .limit(1)\
    .execute()

# Use with your model
X = prepare_features(features.data[0]['features'])
prediction = model.predict(X)
```

### Historical Analysis
```sql
-- Analyze RSI over time
SELECT 
    timestamp,
    close,
    features->>'rsi_14' as rsi
FROM features
WHERE symbol = 'XAUUSD'
  AND timeframe = '1H'
  AND timestamp > NOW() - INTERVAL '7 days'
ORDER BY timestamp;
```

### Dashboard/Monitoring
- Connect to Supabase from your web app
- Display real-time features
- Create charts and indicators
- Set up alerts based on feature values

---

## 🔧 Configuration

### Update Interval
```bash
# Every 5 minutes (300 seconds)
--interval 300

# Every 15 minutes (900 seconds) [default]
--interval 900

# Every hour (3600 seconds)
--interval 3600
```

### Symbols & Timeframes
```bash
# Single
--symbol XAUUSD --timeframe 1H

# Multiple
--symbols XAUUSD,EURUSD,GBPUSD,XAGUSD
--timeframes 15T,30T,1H,4H

# All major forex pairs + metals
--symbols EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,XAUUSD,XAGUSD
--timeframes 1H,4H
```

---

## ❓ Troubleshooting

### "Could not find the 'features' table"
→ Run the SQL schema in Supabase SQL Editor

### "POLYGON_API_KEY not found"
→ Check your `.env` file has the key

### "No data returned for symbol"
→ Market might be closed, try a different symbol

### Script stops running
→ Use screen or nohup to run in background
→ Check logs for errors: `tail -f feature_uploader.log`

---

## ✅ You're Done!

Your system is now:
- ✅ Fetching live data from Polygon every 15 minutes
- ✅ Calculating 264 technical features
- ✅ Uploading to Supabase automatically
- ✅ Available for real-time signal generation

**Next:** Use these features in your trading signals, dashboards, or analysis!

📚 See `SUPABASE_FEATURES_SETUP.md` for advanced configuration.

