#!/usr/bin/env python3
"""Test what Polygon API returns RIGHT NOW"""
import os
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

def test_polygon_fetch():
    """Test fetching current XAUUSD data"""
    symbol = 'XAUUSD'
    ticker = 'C:XAUUSD'

    # Request last 50 bars of 5-minute data
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)

    from_timestamp = int(start_time.timestamp() * 1000)
    to_timestamp = int(end_time.timestamp() * 1000)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{from_timestamp}/{to_timestamp}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}

    print(f"Testing Polygon API for {symbol}")
    print(f"Current UTC time: {end_time}")
    print(f"Requesting from: {start_time}")
    print(f"Requesting to: {end_time}")
    print(f"URL: {url}")
    print()

    response = requests.get(url, params=params, timeout=30)

    print(f"Response status: {response.status_code}")
    print(f"Response text (first 500 chars): {response.text[:500]}")
    print()

    try:
        data = response.json()
    except Exception as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        return

    if 'results' not in data:
        print(f"ERROR: No results in response")
        print(f"Response: {data}")
        return

    print(f"Got {len(data['results'])} bars")

    # Get last 5 bars
    last_5 = data['results'][-5:]
    print("\nLast 5 bars:")
    for bar in last_5:
        bar_time = pd.to_datetime(bar['t'], unit='ms', utc=True)
        print(f"  {bar_time} - Close: {bar['c']}")

    # Check staleness
    last_bar_time = pd.to_datetime(data['results'][-1]['t'], unit='ms', utc=True)
    staleness = end_time - last_bar_time
    print(f"\nLast bar: {last_bar_time}")
    print(f"Current time: {end_time}")
    print(f"Staleness: {staleness}")
    print(f"Staleness in minutes: {staleness.total_seconds() / 60:.1f}")

    # Check if market is open
    now_hour = end_time.hour
    now_dow = end_time.weekday()  # 0=Mon, 4=Fri

    # Forex/metals: Sun 22:00 UTC to Fri 22:00 UTC
    is_weekend = (now_dow == 5 or now_dow == 6 or (now_dow == 4 and now_hour >= 22))
    print(f"\nCurrent day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][now_dow]}")
    print(f"Current hour: {now_hour}:00 UTC")
    print(f"Is weekend/closed: {is_weekend}")

if __name__ == '__main__':
    test_polygon_fetch()
