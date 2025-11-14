#!/usr/bin/env python3
"""
Test Polygon API for live gold/silver data availability
"""
import os
import sys
from datetime import datetime, timedelta, timezone
import requests

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    print("❌ Set POLYGON_API_KEY environment variable")
    print("   Usage: POLYGON_API_KEY=xxx python3 test_polygon_live.py")
    sys.exit(1)

def test_symbol(ticker, symbol_name):
    """Test data freshness for a symbol."""
    print(f"\n{'='*70}")
    print(f"Testing {symbol_name} ({ticker})")
    print('='*70)

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=6)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{start.strftime('%Y-%m-%d')}/{now.strftime('%Y-%m-%d')}"
    params = {
        'adjusted': 'true',
        'sort': 'desc',
        'limit': 50,
        'apiKey': POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text[:200]}")
            return

        data = response.json()
        results = data.get('results', [])

        if not results:
            print("⚠️  No data returned")
            print(f"   Response: {data}")
            return

        # Show latest bar
        latest = results[0]
        latest_time = datetime.fromtimestamp(latest['t'] / 1000, tz=timezone.utc)
        age_minutes = (now - latest_time).total_seconds() / 60

        print(f"Latest bar: {latest_time}")
        print(f"Age: {age_minutes:.1f} minutes ({age_minutes/60:.1f} hours)")
        print(f"Price: ${latest['c']:.2f}")
        print(f"Volume: {latest.get('v', 'N/A')}")
        print(f"Total bars: {len(results)}")

        # Check freshness
        if age_minutes < 10:
            print("✅ Data is FRESH (<10 min old)")
        elif age_minutes < 60:
            print(f"⚠️  Data is slightly stale ({age_minutes:.0f} min old)")
        else:
            print(f"❌ Data is VERY STALE ({age_minutes/60:.1f} hours old)")
            print("   Possible causes:")
            print("   - Markets closed for this symbol")
            print("   - Low trading volume")
            print("   - Polygon API delay/issue")
            print("   - Weekend or maintenance window")

        # Show last 5 bars
        print(f"\nLast 5 bars:")
        for i, bar in enumerate(results[:5]):
            bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=timezone.utc)
            print(f"  {i+1}. {bar_time} | ${bar['c']:.2f} | vol: {bar.get('v', 'N/A')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    now = datetime.now(timezone.utc)
    print(f"Current time: {now}")
    print(f"Day: {now.strftime('%A')}")
    print(f"Hour: {now.hour}:00 UTC")

    # Check market hours
    dow = now.weekday()
    hour = now.hour

    print(f"\nMarket Status:")
    if dow == 4 and hour >= 21:
        print("⚠️  Friday after 21:00 UTC - spot markets closing")
    elif dow == 5:
        print("❌ Saturday - markets closed")
    elif dow == 6 and hour < 17:
        print("❌ Sunday before 17:00 UTC - markets closed")
    else:
        print("✅ Should be open (Sunday 17:00 - Friday 21:00 UTC)")

    # Test both symbols
    test_symbol('C:XAUUSD', 'Gold Spot')
    test_symbol('C:XAGUSD', 'Silver Spot')

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print("If data is stale but markets should be open, possible issues:")
    print("1. Polygon API tier limitations (free tier may have delays)")
    print("2. Polygon data feed issues for spot metals")
    print("3. Low trading volume at this hour (Asian session)")
    print("4. Maintenance window for spot metal feeds")
    print("\nNext steps:")
    print("- Check Polygon status page: https://polygon.io/status")
    print("- Verify API key has real-time access")
    print("- Consider using S3 fallback for 4H timeframes")

if __name__ == '__main__':
    main()
