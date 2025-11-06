# 🚀 Supabase Real-Time Features Setup

Complete guide to automatically upload calculated features to Supabase.

---

## 📋 Overview

This system:
1. ✅ Fetches live OHLCV data from Polygon API
2. ✅ Calculates all 246+ technical indicators and features
3. ✅ Uploads to Supabase `features` table
4. ✅ Runs continuously on a schedule
5. ✅ Provides real-time features for signal generation

---

## 🔧 Step 1: Create Supabase Table

### Option A: Using Supabase Dashboard

1. Go to your Supabase project: https://ifetofkhyblyijghuwzs.supabase.co
2. Navigate to **SQL Editor**
3. Copy the contents of `supabase_schema_features.sql`
4. Paste and run the SQL script
5. ✅ Table `features` is now created!

### Option B: Using Supabase CLI

```bash
# Install Supabase CLI (if not installed)
brew install supabase/tap/supabase

# Login to Supabase
supabase login

# Link your project
supabase link --project-ref ifetofkhyblyijghuwzs

# Run the migration
supabase db push --file supabase_schema_features.sql
```

---

## ⚙️ Step 2: Verify Environment Variables

Make sure your `.env` file has:

```bash
# Polygon API
POLYGON_API_KEY=jVLDXLylHzIpygLbXc0oYuuMGKnNOqpx

# Supabase
SUPABASE_URL=https://ifetofkhyblyijghuwzs.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlmZXRvZmtoeWJseWlqZ2h1d3pzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQyNTIyNTEsImV4cCI6MjA2OTgyODI1MX0.nOzUHck9fqOxvOHPOY8FE2YzmVAX1cohmb64wS9J5MQ
```

✅ Already configured!

---

## 🚀 Step 3: Run the Feature Uploader

### Single Upload (Test)

Upload features for XAUUSD 1H once:

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

python upload_features_to_supabase.py --symbol XAUUSD --timeframe 1H
```

### Multiple Symbols/Timeframes

Upload for multiple combinations:

```bash
python upload_features_to_supabase.py \
    --symbols XAUUSD,EURUSD,GBPUSD \
    --timeframes 1H,4H
```

### Continuous Mode (Recommended)

Run continuously, updating every 15 minutes:

```bash
python upload_features_to_supabase.py \
    --symbols XAUUSD,EURUSD \
    --timeframes 1H,4H \
    --continuous \
    --interval 900
```

This will:
- ✅ Update features every 15 minutes (900 seconds)
- ✅ Process both symbols and timeframes
- ✅ Log everything to `feature_uploader.log`
- ✅ Run indefinitely until stopped (Ctrl+C)

---

## 📊 Step 4: Query Features from Supabase

### Using Python

```python
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

# Get latest features for XAUUSD 1H
response = supabase.table('features')\
    .select('*')\
    .eq('symbol', 'XAUUSD')\
    .eq('timeframe', '1H')\
    .order('timestamp', desc=True)\
    .limit(1)\
    .execute()

latest = response.data[0]
print(f"Latest XAUUSD 1H data:")
print(f"  Timestamp: {latest['timestamp']}")
print(f"  Close: {latest['close']}")
print(f"  Features: {len(latest['features'])} calculated")
print(f"  RSI: {latest['features']['rsi_14']}")
print(f"  MACD: {latest['features']['macd']}")
```

### Using SQL in Supabase Dashboard

```sql
-- Get latest features for all symbols
SELECT * FROM public.latest_features;

-- Get XAUUSD 1H data for last 24 hours
SELECT 
    timestamp,
    close,
    features->>'rsi_14' as rsi,
    features->>'macd' as macd,
    features->>'atr' as atr
FROM public.features
WHERE symbol = 'XAUUSD' 
    AND timeframe = '1H'
    AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Count features per symbol
SELECT 
    symbol,
    timeframe,
    COUNT(*) as record_count,
    MAX(timestamp) as latest_timestamp
FROM public.features
GROUP BY symbol, timeframe;
```

---

## 🔄 Step 5: Run as Background Service

### Option A: Using systemd (Linux/Server)

Create `/etc/systemd/system/feature-uploader.service`:

```ini
[Unit]
Description=Trading Feature Uploader
After=network.target

[Service]
Type=simple
User=omar
WorkingDirectory=/Users/omar/Desktop/ML_Trading
Environment="PATH=/Users/omar/Desktop/ML_Trading/.venv/bin"
ExecStart=/Users/omar/Desktop/ML_Trading/.venv/bin/python upload_features_to_supabase.py --symbols XAUUSD,EURUSD --timeframes 1H,4H --continuous --interval 900
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable feature-uploader
sudo systemctl start feature-uploader
sudo systemctl status feature-uploader
```

### Option B: Using screen (Quick)

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

# Start in detached screen session
screen -dmS feature-uploader python upload_features_to_supabase.py \
    --symbols XAUUSD,EURUSD \
    --timeframes 1H,4H \
    --continuous

# Check if running
screen -ls

# Attach to see output
screen -r feature-uploader

# Detach: Ctrl+A, then D
```

### Option C: Using nohup (Background)

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

nohup python upload_features_to_supabase.py \
    --symbols XAUUSD,EURUSD \
    --timeframes 1H,4H \
    --continuous \
    --interval 900 > feature_uploader_output.log 2>&1 &

# Get process ID
echo $!

# Check if running
ps aux | grep upload_features_to_supabase

# Stop (replace PID with actual process ID)
kill <PID>
```

### Option D: Using cron (Scheduled)

Run every 15 minutes via cron:

```bash
# Edit crontab
crontab -e

# Add this line (runs every 15 minutes)
*/15 * * * * cd /Users/omar/Desktop/ML_Trading && .venv/bin/python upload_features_to_supabase.py --symbols XAUUSD,EURUSD --timeframes 1H,4H >> /Users/omar/Desktop/ML_Trading/cron_features.log 2>&1
```

---

## 📈 Step 6: Monitor & Debug

### Check Logs

```bash
# Real-time log monitoring
tail -f feature_uploader.log

# Last 50 lines
tail -50 feature_uploader.log

# Search for errors
grep ERROR feature_uploader.log
```

### Check Supabase Table

```bash
# Count records
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
count = supabase.table('features').select('*', count='exact').execute()
print(f'Total records: {count.count}')
"
```

---

## 🎯 Example Workflow

```bash
# 1. Create Supabase table (one-time)
# Run supabase_schema_features.sql in Supabase dashboard

# 2. Test single upload
python upload_features_to_supabase.py --symbol XAUUSD --timeframe 1H

# 3. If successful, run continuously
python upload_features_to_supabase.py \
    --symbols XAUUSD,EURUSD,GBPUSD,XAGUSD \
    --timeframes 30T,1H,4H \
    --continuous \
    --interval 900

# 4. Monitor logs in another terminal
tail -f feature_uploader.log

# 5. Query data from Supabase
# Use Python or SQL examples above
```

---

## 🔧 Troubleshooting

### Error: "POLYGON_API_KEY not found"
- Check `.env` file has `POLYGON_API_KEY=...`
- Run from ML_Trading directory

### Error: "SUPABASE_URL not found"
- Check `.env` file has `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`
- Make sure quotes are removed from values

### Error: "relation 'features' does not exist"
- Run `supabase_schema_features.sql` in Supabase SQL Editor
- Check table was created successfully

### Error: "No data returned"
- Market might be closed
- Check Polygon API key is valid
- Try a different symbol

### Performance is slow
- Reduce number of symbols/timeframes
- Increase update interval
- Check network connection

---

## 📊 Data Structure

### features table columns:

```
id                BIGSERIAL (primary key)
symbol            TEXT
timeframe         TEXT
timestamp         TIMESTAMPTZ
open              DOUBLE PRECISION
high              DOUBLE PRECISION
low               DOUBLE PRECISION
close             DOUBLE PRECISION
volume            DOUBLE PRECISION
features          JSONB (all calculated features)
feature_count     INTEGER
data_source       TEXT
created_at        TIMESTAMPTZ
updated_at        TIMESTAMPTZ
```

### features JSONB contains:

- 180+ TA-Lib indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Custom regime features
- Volatility features
- Volume features
- Pattern recognition
- Statistical features

---

## 🚀 Production Deployment

For production use:

1. **Use a VPS/Cloud Server** (DigitalOcean, AWS, GCP)
2. **Set up systemd service** for automatic restart
3. **Configure monitoring** (e.g., Sentry, DataDog)
4. **Add alerting** for failures
5. **Use environment-specific configs**
6. **Implement rate limiting** for Polygon API
7. **Add data validation** before upload
8. **Set up backup/recovery**

---

## 📝 Next Steps

1. ✅ Create Supabase table
2. ✅ Test single upload
3. ✅ Run continuous mode
4. ✅ Monitor logs
5. ✅ Query features from your app
6. ✅ Deploy to production

Your real-time feature pipeline is ready! 🎉

