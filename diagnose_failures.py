#!/usr/bin/env python3
"""
Diagnose why signal generation and auditing are failing.
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

print("=" * 80)
print("SIGNAL GENERATION FAILURE DIAGNOSTICS")
print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

issues = []
warnings = []

# Test 1: Environment Variables
print("\n1. ENVIRONMENT VARIABLES")
print("-" * 80)

env_vars = {
    'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY'),
    'SUPABASE_URL': os.getenv('SUPABASE_URL'),
    'SUPABASE_KEY': os.getenv('SUPABASE_KEY'),
}

for key, value in env_vars.items():
    if value:
        print(f"   ✅ {key}: Set")
    else:
        issues.append(f"{key} not set in environment")
        print(f"   ❌ {key}: NOT SET")

# Test 2: Check if it's market hours
print("\n2. MARKET HOURS CHECK")
print("-" * 80)

now = datetime.now(timezone.utc)
day = now.weekday()
hour = now.hour

print(f"   Current: {now.strftime('%A %H:%M UTC')}")

if day == 6 and hour < 22:  # Sunday before 22:00
    warnings.append("Markets closed (Sunday before 22:00 UTC)")
    print(f"   ⚠️  Markets are CLOSED (Sunday before 22:00)")
elif day == 5 and hour >= 22:  # Friday after 22:00
    warnings.append("Markets closed (Weekend)")
    print(f"   ⚠️  Markets are CLOSED (Weekend)")
else:
    print(f"   ✅ Markets should be OPEN")

# Test 3: Polygon API Test
print("\n3. POLYGON API TEST")
print("-" * 80)

api_key = os.getenv('POLYGON_API_KEY')
if not api_key:
    issues.append("Cannot test Polygon API - no API key")
    print(f"   ❌ Skipped - no API key")
else:
    try:
        import requests

        # Test with EURUSD (simple pair)
        ticker = 'C:EURUSD'
        end_date = now
        start_date = end_date - timedelta(days=1)

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50, 'apiKey': api_key}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results_count = len(data.get('results', []))
            if results_count > 0:
                latest = data['results'][-1]
                latest_time = datetime.fromtimestamp(latest['t'] / 1000, tz=timezone.utc)
                age = now - latest_time

                print(f"   ✅ Polygon API: Working")
                print(f"      Latest bar: {latest_time.strftime('%Y-%m-%d %H:%M UTC')}")
                print(f"      Age: {age.total_seconds() / 60:.1f} minutes")

                if age.total_seconds() > 3600:  # > 1 hour
                    warnings.append(f"Polygon data is {age.total_seconds() / 60:.0f} minutes old")
                    print(f"      ⚠️  Data is stale!")
            else:
                issues.append("Polygon API returned no data")
                print(f"   ❌ Polygon API: No data returned")
        elif response.status_code == 403:
            issues.append("Polygon API: 403 Forbidden (check API key or plan limits)")
            print(f"   ❌ Polygon API: 403 Forbidden")
            print(f"      Check: API key valid? Plan has required permissions?")
        elif response.status_code == 429:
            issues.append("Polygon API: 429 Rate Limited")
            print(f"   ❌ Polygon API: 429 Rate Limited")
        else:
            issues.append(f"Polygon API: HTTP {response.status_code}")
            print(f"   ❌ Polygon API: HTTP {response.status_code}")
            print(f"      Response: {response.text[:200]}")

    except Exception as e:
        issues.append(f"Polygon API test failed: {str(e)[:100]}")
        print(f"   ❌ Error: {str(e)[:100]}")

# Test 4: Supabase Connection
print("\n4. SUPABASE CONNECTION TEST")
print("-" * 80)

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')

if not url or not key:
    issues.append("Cannot test Supabase - missing credentials")
    print(f"   ❌ Skipped - no credentials")
else:
    try:
        from supabase import create_client

        supabase = create_client(url, key)

        # Try to read from live_signals table
        response = supabase.table('live_signals').select('*').limit(1).execute()

        print(f"   ✅ Supabase: Connected")

        # Check recent signals
        since = (now - timedelta(hours=1)).isoformat()
        response = supabase.table('live_signals').select('symbol,timeframe,timestamp').gte('timestamp', since).order('timestamp', desc=True).limit(10).execute()

        if response.data:
            latest = response.data[0]
            latest_time = datetime.fromisoformat(latest['timestamp'].replace('Z', '+00:00'))
            age = now - latest_time

            print(f"      Latest signal: {latest['symbol']} {latest['timeframe']}")
            print(f"      Timestamp: {latest_time.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"      Age: {age.total_seconds() / 60:.1f} minutes")

            if age.total_seconds() > 600:  # > 10 minutes
                warnings.append(f"Latest signal is {age.total_seconds() / 60:.0f} minutes old")
                print(f"      ⚠️  Signals are stale!")
        else:
            warnings.append("No signals found in last hour")
            print(f"      ⚠️  No signals in last hour")

    except Exception as e:
        issues.append(f"Supabase connection failed: {str(e)[:100]}")
        print(f"   ❌ Error: {str(e)[:100]}")

# Test 5: Model Loading
print("\n5. MODEL LOADING TEST")
print("-" * 80)

try:
    from balanced_model import BalancedModel
    import __main__
    setattr(__main__, 'BalancedModel', BalancedModel)

    from ensemble_predictor import EnsemblePredictor

    test_symbols = ['XAUUSD', 'XAGUSD']
    for symbol in test_symbols:
        try:
            predictor = EnsemblePredictor(symbol)
            if len(predictor.models) > 0:
                print(f"   ✅ {symbol}: {len(predictor.models)} models loaded")
            else:
                issues.append(f"{symbol}: No models loaded")
                print(f"   ❌ {symbol}: No models loaded")
        except Exception as e:
            issues.append(f"{symbol} model loading failed: {str(e)[:100]}")
            print(f"   ❌ {symbol}: {str(e)[:100]}")

except Exception as e:
    issues.append(f"Model loading test failed: {str(e)[:100]}")
    print(f"   ❌ Error: {str(e)[:100]}")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

print(f"\nCritical Issues: {len(issues)}")
print(f"Warnings: {len(warnings)}")

if issues:
    print("\n❌ CRITICAL ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")

if warnings:
    print("\n⚠️  WARNINGS:")
    for i, warn in enumerate(warnings, 1):
        print(f"   {i}. {warn}")

print("\n" + "=" * 80)
print("LIKELY ROOT CAUSE")
print("=" * 80)

if "POLYGON_API_KEY" in str(issues) or "SUPABASE" in str(issues):
    print("\n❌ ENVIRONMENT ISSUE")
    print("   Missing required API keys in GitHub Actions secrets")
    print("   Fix: Add POLYGON_API_KEY, SUPABASE_URL, SUPABASE_KEY to repo secrets")
elif "403" in str(issues) or "Rate Limited" in str(issues):
    print("\n❌ API ACCESS ISSUE")
    print("   Polygon API key invalid or plan limitations")
    print("   Fix: Check API key validity and subscription plan")
elif "Markets closed" in str(warnings) or "stale" in str(warnings):
    print("\n⚠️  TIMING ISSUE")
    print("   Markets are closed or data is stale")
    print("   This is normal outside trading hours")
    print("   Signals will resume when markets reopen")
elif issues:
    print("\n❌ TECHNICAL ISSUE")
    print("   Check the errors above for specific failures")
    print("   Review GitHub Actions workflow logs for details")
else:
    print("\n✅ NO CRITICAL ISSUES DETECTED")
    print("   System appears functional")
    print("   Check GitHub Actions logs for specific error messages")

print("=" * 80)

if issues:
    sys.exit(1)
