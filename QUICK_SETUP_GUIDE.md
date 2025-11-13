# Quick Setup Guide - Generate Signals

## What You Need

To generate trading signals, you need **2 things**:

1. **Polygon API** - For real-time market data
2. **Supabase** - For storing signals in a database

## Step-by-Step Setup

### 1. Get Polygon API Key (5 minutes)

```bash
1. Go to: https://polygon.io/
2. Sign up (free tier is enough to start)
3. Go to: Dashboard → API Keys
4. Copy your API key
```

### 2. Get Supabase Credentials (5 minutes)

```bash
1. Go to: https://supabase.com/
2. Sign up (free tier gives you 500MB database)
3. Create a new project (takes 2-3 minutes)
4. Go to: Settings → API
5. Copy these 3 values:
   - Project URL
   - anon public key
   - service_role key (expand "service_role" section)
```

### 3. Configure .env File

Open `.env` file and replace the placeholders:

```bash
# Replace these values:
POLYGON_API_KEY=paste_your_polygon_key_here
SUPABASE_URL=paste_your_supabase_url_here
SUPABASE_KEY=paste_your_anon_key_here
SUPABASE_SERVICE_KEY=paste_your_service_role_key_here
```

### 4. Create Database Tables (1 minute)

Run this SQL in Supabase:

```bash
# Option 1: Via Supabase Dashboard
1. Go to your Supabase project
2. Click "SQL Editor" in left menu
3. Copy contents of: supabase_schema.sql
4. Paste and click "Run"

# Option 2: Via command line
cat supabase_schema.sql | pbcopy  # Copy to clipboard
# Then paste in Supabase SQL Editor
```

This creates 4 tables:
- `ml_models` - Stores your trained models
- `live_signals` - Stores generated trading signals
- `trades` - Stores executed trades
- `performance_metrics` - Stores performance stats

### 5. Test Your Setup

```bash
# Test Polygon API connection
python test_guardrails_live.py

# Should see:
# ✅ Received 50000 bars
# ✅ ALL GUARDRAILS PASSED
```

## Now Generate Signals! 🚀

```bash
# Generate signals for all symbols
python signal_generator.py

# Or run live trading engine
python live_trading_engine.py
```

## What Gets Stored in Supabase

Every time you generate signals, this data is saved:

```
live_signals table:
- Symbol (XAUUSD, XAGUSD, etc.)
- Signal type (long/short/flat)
- Confidence (0-100%)
- Entry price
- Stop loss
- Take profit
- Timestamp
```

You can view your signals in Supabase Dashboard → Table Editor → live_signals

## Troubleshooting

**Error: POLYGON_API_KEY not found**
→ Check your .env file, make sure no spaces around `=`

**Error: SUPABASE_URL not found**
→ Check your .env file has correct values

**Error: relation "live_signals" does not exist**
→ You didn't run the SQL schema. Go back to Step 4.

**Free tier limits:**
- Polygon: 5 API calls/minute (enough for testing)
- Supabase: 500MB database (enough for 100,000+ signals)

**Need unlimited?**
- Polygon Premium: $199/month (unlimited API calls)
- Supabase Pro: $25/month (8GB database)

## Files Overview

```
.env                          ← Your secrets (NEVER commit to git)
signal_generator.py           ← Generates signals for all symbols
live_trading_engine.py        ← Runs live trading loop
test_guardrails_live.py       ← Tests API connection
supabase_schema.sql           ← Database schema (run once)
market_costs.py               ← TP/SL parameters
execution_guardrails.py       ← Safety filters
```

## Next Steps

Once signals are generating:

1. **Monitor signals** - Check Supabase dashboard
2. **Backtest fixes** - Run `validate_backtest_with_costs.py`
3. **Deploy to cloud** - See RUNBOOK.md for deployment guide

---

**Questions?** Check POSTMORTEM.md for detailed explanations of all fixes.
