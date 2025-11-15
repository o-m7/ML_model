#!/usr/bin/env python3
"""
Diagnose Polygon API Data Freshness
Tests if Polygon is returning current data or stale historical data
"""

import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

def test_polygon_freshness():
    """Test Polygon API to see what data it's actually returning"""

    print("="*80)
    print("POLYGON API DATA FRESHNESS DIAGNOSTIC")
    print("="*80)

    now = datetime.now(timezone.utc)
    print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Day of week: {now.strftime('%A')} (0=Mon, 4=Fri, 5=Sat, 6=Sun: {now.weekday()})")

    # Check market status
    if now.weekday() == 5:
        print("⚠️  SATURDAY - Market is CLOSED")
    elif now.weekday() == 6 and now.hour < 22:
        print("⚠️  SUNDAY (before 22:00) - Market is CLOSED")
    elif now.weekday() == 4 and now.hour >= 22:
        print("⚠️  FRIDAY (after 22:00) - Market is CLOSING")
    else:
        print("✅ Market should be OPEN")

    symbols = [
        ('C:XAUUSD', 'XAUUSD'),
        ('C:XAGUSD', 'XAGUSD'),
    ]

    timeframes = [
        (1, 'minute', '1min'),
        (5, 'minute', '5T'),
        (15, 'minute', '15T'),
        (60, 'minute', '1H'),
    ]

    print("\n" + "="*80)
    print("TESTING POLYGON API RESPONSES")
    print("="*80)

    for ticker, symbol in symbols:
        print(f"\n{symbol} ({ticker}):")
        print("-"*80)

        for multiplier, timespan, label in timeframes:
            # Request last 5 bars
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=2)

            from_ts = int(start_time.timestamp() * 1000)
            to_ts = int(end_time.timestamp() * 1000)

            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_ts}/{to_ts}"

            params = {
                'adjusted': 'true',
                'sort': 'desc',
                'limit': 5,
                'apiKey': POLYGON_API_KEY
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()

                if 'results' in data and data['results']:
                    latest = data['results'][0]
                    bar_time = datetime.fromtimestamp(latest['t'] / 1000, tz=timezone.utc)
                    age_seconds = (now - bar_time).total_seconds()
                    age_minutes = age_seconds / 60
                    age_hours = age_minutes / 60

                    status = "✅" if age_hours < 1 else "⚠️" if age_hours < 24 else "❌"

                    print(f"  {status} {label:>6}: Latest bar {bar_time.strftime('%Y-%m-%d %H:%M UTC')} "
                          f"(age: {age_hours:.1f}h / {age_minutes:.0f}min)")

                    if age_hours > 2:
                        print(f"       ⚠️  DATA IS STALE! Should be <1 hour during market hours")
                else:
                    print(f"  ❌ {label:>6}: No data returned")
                    if 'status' in data:
                        print(f"       API Status: {data.get('status')}")
                        print(f"       Message: {data.get('message', 'No message')}")

            except Exception as e:
                print(f"  ❌ {label:>6}: Error - {e}")

    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    print("\nPossible causes if data is stale:")
    print("  1. API subscription level (delayed vs real-time)")
    print("  2. Market is closed (weekend/holiday)")
    print("  3. API rate limiting or throttling")
    print("  4. Data provider issues")
    print("  5. Network connectivity issues")

    print("\nTo fix:")
    print("  - Check Polygon subscription (should be real-time, not delayed)")
    print("  - Verify API key has correct permissions")
    print("  - Check if today is a market holiday")
    print("  - Try different symbols to rule out symbol-specific issues")

    print("\n" + "="*80)

if __name__ == "__main__":
    if not POLYGON_API_KEY:
        print("❌ POLYGON_API_KEY not found in environment")
        print("   Set it in .env file")
        exit(1)

    test_polygon_freshness()
