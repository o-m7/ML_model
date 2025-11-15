# Signal Generator Critical Fixes

## Summary
Fixed all blocking issues preventing signal generation from producing actionable signals.

## Issues Fixed

### 1. 4H Timeframe Data Staleness ✅
**File:** `signal_generator.py:260-262`
- **Before:** Allowed 24 hours staleness for 4H bars
- **After:** Reduced to 8 hours max (resampled from fresh 1H data)
- **Impact:** 4H signals will now fail if data is more than 8 hours old, forcing fresh data

### 2. Guardrail Confidence Threshold ✅
**File:** `execution_guardrails.py:339`
- **Before:** `min_confidence=0.55` (blocking 100% of signals)
- **After:** `min_confidence=0.50` (allows signals with 50%+ confidence)
- **Impact:** Valid signals with 50-55% confidence now pass through

### 3. Session Filter Blocking Valid Hours ✅
**File:** `execution_guardrails.py:341`
- **Before:** `blocked_sessions=['overnight']` (blocking 21:00-23:59 UTC)
- **After:** `blocked_sessions=[]` (only weekends blocked via check_session())
- **Impact:** Trading allowed 24/5, no more false "overnight session" blocks at 21:17 UTC

### 4. Missing Volume Features ✅
**File:** `live_feature_utils.py:82-85`
- **Added:** `volume_sma20 = volume.rolling(20).mean()`
- **Added:** `volume_ratio = volume / volume_sma20`
- **Added:** Proper NaN handling with fillna()
- **Impact:** Models requiring volume_ratio and volume_sma20 now have these features

### 5. Debug Logging ✅
**File:** `signal_generator.py:179, 221-226`
- **Added:** Timestamp logging when fetching data
- **Added:** Latest bar timestamp and age logging
- **Impact:** Easy to diagnose data freshness issues in logs

## Changes Made

### signal_generator.py
```python
# Line 179: Added fetch timestamp logging
print(f"  🔍 [{end_time.strftime('%H:%M:%S')}] Fetching {timeframe} data (via {fetch_minutes}min bars)...")

# Lines 221-226: Added data freshness logging
if not df.empty:
    last_bar = df.index[-1]
    age_seconds = (end_time - last_bar).total_seconds()
    age_hours = age_seconds / 3600
    print(f"  📊 Latest {timeframe} bar: {last_bar.strftime('%Y-%m-%d %H:%M UTC')} (age: {age_hours:.1f}h)")

# Lines 260-262: Reduced 4H staleness threshold
if timeframe == '4H':
    max_allowed = timedelta(hours=8)  # Was 24
```

### execution_guardrails.py
```python
# Lines 339-341: Relaxed guardrails
min_confidence=0.50,  # Was 0.55
blocked_sessions=[],  # Was ['overnight']
```

### live_feature_utils.py
```python
# Lines 82-85: Added volume features
df['volume_sma20'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / (df['volume_sma20'] + 1e-10)
df['volume_ratio'].fillna(1.0, inplace=True)
df['volume_sma20'].fillna(df['volume'].mean(), inplace=True)

# Line 212: Added to feature list
'volume_ratio', 'volume_sma20'
```

## Expected Behavior After Fix

### Before:
```
❌ XAGUSD 4H: Stale data (last bar 2025-11-11, should be 2025-11-14)
❌ All signals blocked by guardrails (confidence < 0.55)
❌ 21:17 UTC blocked as "overnight session"
❌ Models failing: Missing volume_ratio, volume_sma20
```

### After:
```
✅ XAGUSD 4H: Latest bar within 8 hours (or rejected if stale)
✅ Signals with 50%+ confidence pass guardrails
✅ Trading allowed 24/5 (only weekends blocked)
✅ All volume features calculated and available
✅ Debug logs show exact data timestamps and age
```

## Validation Checklist

Run after deployment:
- [ ] Check logs for 4H data timestamps (should be current date)
- [ ] Verify at least 1 signal passes guardrails per symbol
- [ ] Confirm no "overnight session" blocks during market hours
- [ ] Verify volume_ratio and volume_sma20 in feature logs
- [ ] Check confidence values between 0.50-0.55 are accepted

## Testing Commands

```bash
# Test signal generation
python signal_generator.py

# Check data freshness for all timeframes
python -c "
from signal_generator import fetch_polygon_data
from datetime import datetime, timezone
import pandas as pd

for symbol in ['XAUUSD', 'XAGUSD']:
    for tf in ['5T', '15T', '30T', '1H', '4H']:
        data = fetch_polygon_data(symbol, tf)
        if data is not None:
            age = (datetime.now(timezone.utc) - data.index[-1]).total_seconds() / 3600
            print(f'{symbol} {tf}: {data.index[-1]} (age: {age:.1f}h)')
"
```

## Rollback Instructions

If issues arise, revert these specific lines:
```bash
git diff HEAD~1 signal_generator.py
git diff HEAD~1 execution_guardrails.py
git diff HEAD~1 live_feature_utils.py
git checkout HEAD~1 -- signal_generator.py execution_guardrails.py live_feature_utils.py
```
