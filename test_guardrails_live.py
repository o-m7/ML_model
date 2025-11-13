#!/usr/bin/env python3
"""
LIVE GUARDRAILS TEST - WITH REAL POLYGON DATA
==============================================

Demonstrates execution guardrails using LIVE data from Polygon REST API.

This fetches 50,000 bars (default) for accurate pattern detection and feature calculation.

Features:
- Fetches 50,000 bars of historical + real-time data from Polygon
- Calculates ATR and other indicators on live market data
- Runs all 6 execution guardrail checks on actual market conditions
- Shows current spread, volatility, and data staleness

Usage:
    python test_guardrails_live.py --api-key YOUR_API_KEY
    python test_guardrails_live.py --api-key YOUR_API_KEY --symbol XAUUSD --tf 15T
    python test_guardrails_live.py --symbol XAGUSD --tf 5T  # Uses .env file

Get your free API key at: https://polygon.io/

Note: Free tier limited to 5 API calls/minute. Consider Premium for unlimited calls.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Import our guardrails
from execution_guardrails import ExecutionGuardrails, get_moderate_guardrails
from market_costs import get_costs

# Load environment variables
load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

# Ticker mapping
TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD',
}

TIMEFRAME_MINUTES = {
    '5T': 5,
    '15T': 15,
    '30T': 30,
    '1H': 60,
    '4H': 240
}


def fetch_live_data(symbol: str, timeframe: str, bars: int = 50000):
    """
    Fetch live data from Polygon REST API.

    Args:
        symbol: Trading symbol (XAUUSD, XAGUSD, etc.)
        timeframe: Timeframe (5T, 15T, 30T, 1H, 4H)
        bars: Number of bars to fetch (default: 50,000 for pattern detection)

    Returns:
        DataFrame with OHLCV data
    """
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not found in environment. Set it in .env file.")

    ticker = TICKER_MAP.get(symbol, symbol)
    minutes = TIMEFRAME_MINUTES[timeframe]

    # Calculate date range
    # 50,000 bars needs ~60-90 days depending on timeframe (accounting for weekends/holidays)
    from datetime import timedelta
    end_time = datetime.now(timezone.utc)
    days_needed = max(60, int(bars * minutes / (60 * 24)) + 30)  # Extra buffer for market gaps
    start_time = end_time - timedelta(days=days_needed)

    # Build API request
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{minutes}/minute"
    url += f"/{start_time.strftime('%Y-%m-%d')}/{end_time.strftime('%Y-%m-%d')}"

    params = {
        'adjusted': 'true',
        'sort': 'desc',
        'limit': min(bars, 50000),  # Polygon API max is 50,000
        'apiKey': POLYGON_API_KEY
    }

    print(f"📡 Fetching live {symbol} data from Polygon...")
    print(f"   Timeframe: {timeframe} ({minutes} minutes)")
    print(f"   Requesting {bars} bars (~{bars*minutes/1440:.1f} trading days)")
    print(f"   Date range: {start_time.date()} to {end_time.date()}")

    try:
        response = requests.get(url, params=params, timeout=30)  # Increased timeout for large requests
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'OK' or not data.get('results'):
            print(f"❌ No data returned from Polygon")
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message', 'No message')}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"✅ Received {len(df)} bars")
        print(f"   First bar: {df.iloc[0]['timestamp']}")
        print(f"   Latest bar: {df.iloc[-1]['timestamp']}")
        print(f"   Current price: ${df.iloc[-1]['close']:.2f}")

        # Calculate data span
        data_span_days = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).days
        print(f"   Data span: {data_span_days} days ({len(df)} bars)")

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return None


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR (Average True Range)."""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is simple moving average of TR
    atr = tr.rolling(window=period).mean()

    return atr


def estimate_spread(symbol: str, current_price: float) -> float:
    """
    Estimate current spread.

    In production, you'd get this from broker's real-time quotes.
    For demo, we use typical spreads from market_costs.py
    """
    costs = get_costs(symbol)

    # Convert pip spread to price
    if symbol == 'XAUUSD':
        spread = costs.spread_pips * 0.10  # 1 pip = $0.10 for gold
    elif symbol == 'XAGUSD':
        spread = costs.spread_pips * 0.01  # 1 pip = $0.01 for silver
    else:  # Forex
        spread = costs.spread_pips * 0.0001  # 1 pip = 0.0001 for forex

    return spread


def test_guardrails_with_live_data(symbol: str = 'XAUUSD', timeframe: str = '15T', bars: int = 50000):
    """
    Test guardrails using live Polygon data.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        bars: Number of bars to fetch (default: 50,000 for pattern detection)
    """
    print("\n" + "="*80)
    print(f"LIVE GUARDRAILS TEST - {symbol} {timeframe}")
    print("="*80 + "\n")

    # Fetch live data
    df = fetch_live_data(symbol, timeframe, bars=bars)

    if df is None or len(df) == 0:
        print("\n❌ Cannot proceed without data")
        return 1

    # Calculate ATR
    print("\n📊 Calculating technical indicators...")
    df['atr14'] = calculate_atr(df, period=14)

    # Get latest bar
    latest_bar = df.iloc[-1]
    current_price = latest_bar['close']
    atr = latest_bar['atr14']
    last_bar_time = latest_bar['timestamp']

    # Check for NaN ATR
    if pd.isna(atr):
        print("⚠️  ATR is NaN (not enough bars), using 2% of price as estimate")
        atr = current_price * 0.02

    # Estimate spread
    spread = estimate_spread(symbol, current_price)

    # Calculate data staleness
    now = datetime.now(timezone.utc)
    if last_bar_time.tz is None:
        last_bar_time = last_bar_time.tz_localize('UTC')

    staleness_seconds = (now - last_bar_time).total_seconds()
    staleness_minutes = staleness_seconds / 60

    print(f"✅ Latest market data:")
    print(f"   Timestamp: {last_bar_time}")
    print(f"   Price: ${current_price:.2f}")
    print(f"   ATR(14): ${atr:.2f}")
    print(f"   Spread: ${spread:.2f}")
    print(f"   Data age: {staleness_minutes:.1f} minutes")

    # Mock confidence and latency (in production, these come from your model)
    confidence = 0.62  # Example: 62% confidence
    latency_ms = 150.0  # Example: 150ms latency

    print(f"\n📈 Trading parameters:")
    print(f"   Model confidence: {confidence:.1%}")
    print(f"   Execution latency: {latency_ms:.0f}ms")

    # Create guardrails
    print(f"\n🛡️  Initializing execution guardrails...")
    guards = get_moderate_guardrails()

    print(f"   Max data age: {guards.max_data_age_seconds}s")
    print(f"   Max spread/ATR: {guards.max_spread_atr_ratio:.1%}")
    print(f"   Min confidence: {guards.min_confidence:.1%}")
    print(f"   Max latency: {guards.max_latency_ms:.0f}ms")
    print(f"   Volatility range: {guards.min_atr_pct:.2%} - {guards.max_atr_pct:.1%}")

    # Run guardrail checks
    print(f"\n{'='*80}")
    print("GUARDRAIL CHECKS - LIVE DATA")
    print("="*80 + "\n")

    timeframe_minutes = TIMEFRAME_MINUTES[timeframe]

    results = guards.check_all(
        last_bar_time=last_bar_time,
        timeframe_minutes=timeframe_minutes,
        current_spread=spread,
        atr=atr,
        price=current_price,
        confidence=confidence,
        latency_ms=latency_ms
    )

    # Display results
    print("Check Results:")
    print("-" * 80)

    for name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"

        # Format based on check type
        if name == 'staleness':
            age_min = result.value / 60 if result.value else 0
            max_min = result.threshold / 60 if result.threshold else 0
            print(f"{status:10s} {name:15s} age={age_min:.1f}min (max {max_min:.1f}min)")
        elif name == 'spread':
            spread_pct = result.value * 100 if result.value else 0
            max_pct = result.threshold * 100 if result.threshold else 0
            print(f"{status:10s} {name:15s} {spread_pct:.1f}% of ATR (max {max_pct:.1f}%)")
        elif name == 'volatility':
            vol_pct = result.value * 100 if result.value else 0
            print(f"{status:10s} {name:15s} ATR={vol_pct:.2f}% of price")
        elif name == 'confidence':
            conf_pct = result.value * 100 if result.value else 0
            min_pct = result.threshold * 100 if result.threshold else 0
            print(f"{status:10s} {name:15s} {conf_pct:.0f}% (min {min_pct:.0f}%)")
        elif name == 'latency':
            print(f"{status:10s} {name:15s} {result.value:.0f}ms (max {result.threshold:.0f}ms)")
        elif name == 'session':
            hour = int(result.value) if result.value else 0
            if 0 <= hour < 8:
                session_name = "Asia"
            elif 8 <= hour < 16:
                session_name = "London"
            elif 13 <= hour < 21:
                session_name = "US/London Overlap"
            else:
                session_name = "Overnight"
            print(f"{status:10s} {name:15s} {session_name} (hour {hour})")
        else:
            print(f"{status:10s} {name:15s} {result.reason if not result.passed else 'OK'}")

    print("\n" + "="*80)

    # Final verdict
    if guards.all_passed(results):
        print("✅ ALL GUARDRAILS PASSED - TRADE ALLOWED")
        print("\nThis trade meets all safety requirements:")
        print("  • Data is fresh and reliable")
        print("  • Spread is acceptable")
        print("  • Volatility is in normal range")
        print("  • Trading session is favorable")
        print("  • Model confidence is sufficient")
        print("  • Execution latency is acceptable")
    else:
        failures = guards.get_failures(results)
        print(f"❌ {len(failures)} GUARDRAILS FAILED - TRADE BLOCKED")
        print("\nReasons:")
        for name, reason in failures.items():
            print(f"  • {name}: {reason}")

    print("="*80 + "\n")

    return 0 if guards.all_passed(results) else 1


def main():
    parser = argparse.ArgumentParser(description='Test guardrails with live Polygon data')
    parser.add_argument('--symbol', type=str, default='XAUUSD',
                       help='Symbol to test (XAUUSD, XAGUSD, EURUSD, etc.)')
    parser.add_argument('--tf', type=str, default='15T',
                       help='Timeframe (5T, 15T, 30T, 1H, 4H)')
    parser.add_argument('--bars', type=int, default=50000,
                       help='Number of bars to fetch (default: 50,000)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Polygon API key (or set POLYGON_API_KEY in .env)')

    args = parser.parse_args()

    # Override global POLYGON_API_KEY if provided via CLI
    global POLYGON_API_KEY
    if args.api_key:
        POLYGON_API_KEY = args.api_key
    elif not POLYGON_API_KEY:
        print("❌ ERROR: Polygon API key not found\n")
        print("Please provide your API key in one of two ways:\n")
        print("  1. Command line:")
        print("     python test_guardrails_live.py --api-key YOUR_API_KEY\n")
        print("  2. Environment variable:")
        print("     Create a .env file with: POLYGON_API_KEY=YOUR_API_KEY")
        print("     Then run: python test_guardrails_live.py\n")
        print("Get your free API key at: https://polygon.io/")
        return 1

    if args.symbol not in TICKER_MAP:
        print(f"❌ Unknown symbol: {args.symbol}")
        print(f"   Available: {', '.join(TICKER_MAP.keys())}")
        return 1

    if args.tf not in TIMEFRAME_MINUTES:
        print(f"❌ Unknown timeframe: {args.tf}")
        print(f"   Available: {', '.join(TIMEFRAME_MINUTES.keys())}")
        return 1

    return test_guardrails_with_live_data(args.symbol, args.tf, args.bars)


if __name__ == '__main__':
    sys.exit(main())
