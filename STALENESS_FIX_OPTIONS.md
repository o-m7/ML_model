# Staleness Issue - Fix Options

**Problem**: Polygon API returning 5+ hour old data, causing signal generation to skip all timeframes.

**Root Cause**: `signal_generator.py` line 262-264 exits early when data exceeds staleness threshold.

---

## Option 1: Increase Staleness Threshold (Quick Fix)

Allow older data temporarily while diagnosing Polygon issue.

**Change in `signal_generator.py` line 256-260:**

```python
# BEFORE (strict):
if timeframe == '4H':
    max_allowed = timedelta(hours=24)
else:
    max_allowed = timedelta(minutes=TIMEFRAME_MINUTES[timeframe] * 2)

# AFTER (relaxed):
if timeframe == '4H':
    max_allowed = timedelta(hours=24)
else:
    # Temporarily allow up to 8 hours staleness for all timeframes
    # This handles Polygon delays and low-volume periods
    max_allowed = timedelta(hours=8)
```

**Pros**: Signals will generate even with delayed Polygon data
**Cons**: Using older data may reduce signal quality
**Use when**: Polygon has known delays or markets have low volume

---

## Option 2: Generate Signals with Stale Data Warning (Moderate)

Still generate signals but flag them as using stale data.

**Change in `signal_generator.py` line 262-264:**

```python
# BEFORE (skip on stale):
if staleness > max_allowed:
    print(f"  ⚠️  {symbol} {timeframe}: Stale data (last bar {last_bar_time} UTC, Δ {staleness})")
    return  # <- Skip signal generation

# AFTER (continue with warning):
stale_data = False
if staleness > max_allowed:
    print(f"  ⚠️  {symbol} {timeframe}: Stale data (last bar {last_bar_time} UTC, Δ {staleness})")
    stale_data = True
    # Continue generating signal but mark it as stale
```

Then add `stale_data` flag to the Supabase payload (line ~370):

```python
supabase_payload = {
    # ... existing fields ...
    'stale_data': stale_data,  # New field
}
```

**Pros**: Still generates signals, tracks data quality
**Cons**: Requires Supabase schema update
**Use when**: Want signals but need to track data freshness

---

## Option 3: Check Polygon API Tier & Status (Diagnostic)

Verify your Polygon API access level and data feed status.

**Steps:**

1. **Check your API tier:**
   - Visit: https://polygon.io/dashboard
   - Free tier: 15-minute delayed data
   - Starter tier: Real-time data

2. **Check Polygon status:**
   - Visit: https://polygon.io/status
   - Look for issues with Forex/Spot data feeds

3. **Test data freshness:**
   ```bash
   POLYGON_API_KEY=xxx python3 test_polygon_live.py
   ```

**If Free Tier**: Upgrade to Starter ($99/month) for real-time data
**If API Issue**: Wait for Polygon to resolve, use Option 1 temporarily

---

## Option 4: Switch to Alternative Data Source (Long-term)

Use different data provider for gold/silver spot prices.

**Alternatives:**
- Alpha Vantage (free tier available)
- OANDA API (forex data)
- Yahoo Finance (delayed but free)

**Pros**: Not dependent on Polygon for spot metals
**Cons**: Requires code changes, may have different data quality

---

## Recommended Immediate Action

**For now, use Option 1** (increase staleness threshold):

```bash
# Apply the fix
git diff signal_generator.py
```

I can make this change if you want signals to generate immediately while we diagnose the Polygon issue.

**Then investigate:**
1. Run `test_polygon_live.py` to see actual data freshness
2. Check your Polygon API tier
3. Check Polygon status page for known issues

---

## Market Hours Note

Gold/Silver spot markets:
- **Open**: Sunday 17:00 UTC - Friday 21:00 UTC
- **Peak liquidity**: London/NY overlap (13:00-17:00 UTC)
- **Lower liquidity**: Asian session (20:00-08:00 UTC)

Current time: Friday 20:16 UTC = **Low liquidity period** (Asian session, near daily close)

This could explain older bars from Polygon if there's simply no trading activity.

---

## Decision Tree

```
Is it after Friday 21:00 UTC or weekend?
├─ YES → Normal (markets closed, stale data expected)
└─ NO → Continue...

Did test_polygon_live.py show fresh data?
├─ YES → Bug in signal_generator logic
└─ NO → Continue...

Is Polygon API tier = Free (15-min delayed)?
├─ YES → Upgrade to Starter OR use Option 1
└─ NO → Continue...

Is Polygon status page showing issues?
├─ YES → Use Option 1 temporarily, wait for fix
└─ NO → Contact Polygon support

Still broken?
└─ Use Option 4 (alternative data source)
```

---

## Want me to apply Option 1 now?

I can increase the staleness threshold to 8 hours so signals generate immediately, then you can investigate the Polygon API issue separately.

Let me know!
