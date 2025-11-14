#!/usr/bin/env python3
"""
Download historical data from Polygon REST API for training.
Fetches 50,000 bars for pattern detection.

Usage:
    python download_training_data.py --symbol XAUUSD --tf 15T
    python download_training_data.py --all
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY or POLYGON_API_KEY == 'your_polygon_api_key_here':
    print("❌ ERROR: Set POLYGON_API_KEY in .env file")
    sys.exit(1)

# Symbols and timeframes for training
SYMBOLS = ['XAUUSD', 'XAGUSD']
TIMEFRAMES = ['5T', '15T', '30T', '1H']

TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD'
}

TF_MINUTES = {
    '5T': 5,
    '15T': 15,
    '30T': 30,
    '1H': 60,
    '4H': 240
}


def fetch_from_polygon(symbol, timeframe, bars_needed=50000):
    """
    Fetch historical bars from Polygon REST API.

    Args:
        symbol: e.g., 'XAUUSD', 'XAGUSD'
        timeframe: '5T', '15T', '30T', '1H'
        bars_needed: Number of bars to fetch (default: 50000 for pattern detection)

    Returns:
        DataFrame with OHLCV data
    """
    ticker = TICKER_MAP.get(symbol, f'C:{symbol}')
    minutes = TF_MINUTES[timeframe]

    # Calculate date range
    # For 50k bars at 5min = 250k minutes = ~174 days
    # For 50k bars at 1H = 50k hours = ~2083 days (~6 years)
    # Use 365 days * 2 = 2 years to be safe
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=730)  # 2 years

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{minutes}/minute"
    url += f"/{start_time.strftime('%Y-%m-%d')}/{end_time.strftime('%Y-%m-%d')}"

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': bars_needed,
        'apiKey': POLYGON_API_KEY
    }

    print(f"  📡 Fetching {symbol} {timeframe} ({bars_needed} bars)...")

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'OK':
            print(f"  ❌ API Error: {data.get('status')}")
            return None

        if not data.get('results'):
            print(f"  ❌ No data returned")
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
        df = df.set_index('timestamp').sort_index()

        print(f"  ✅ Got {len(df):,} bars")
        print(f"     Range: {df.index.min()} to {df.index.max()}")

        return df

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"  ❌ Rate limit exceeded. Wait a minute and try again.")
        else:
            print(f"  ❌ HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def download_data(symbol, timeframe):
    """Download and save data to feature_store."""
    print(f"\n{'='*80}")
    print(f"DOWNLOADING: {symbol} {timeframe}")
    print(f"{'='*80}")

    # Fetch from Polygon
    df = fetch_from_polygon(symbol, timeframe, bars_needed=50000)

    if df is None or df.empty:
        print(f"❌ Failed to download {symbol} {timeframe}")
        return False

    # Save to feature_store
    output_dir = Path(f"feature_store/{symbol}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(output_path)

    print(f"\n💾 Saved: {output_path}")
    print(f"   Bars: {len(df):,}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

    return True


def main():
    parser = argparse.ArgumentParser(description='Download historical data from Polygon')
    parser.add_argument('--symbol', type=str, help='Symbol to download (XAUUSD, XAGUSD, etc.)')
    parser.add_argument('--tf', type=str, help='Timeframe (5T, 15T, 30T, 1H)')
    parser.add_argument('--all', action='store_true', help='Download all symbols and timeframes')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("POLYGON DATA DOWNLOADER - 50,000 BARS FOR PATTERN DETECTION")
    print("="*80 + "\n")

    success = []
    failed = []

    if args.all:
        # Download all
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                if download_data(symbol, timeframe):
                    success.append((symbol, timeframe))
                else:
                    failed.append((symbol, timeframe))

    elif args.symbol and args.tf:
        # Download specific
        if download_data(args.symbol, args.tf):
            success.append((args.symbol, args.tf))
        else:
            failed.append((args.symbol, args.tf))

    else:
        print("❌ Please specify --symbol and --tf, or use --all")
        parser.print_help()
        return 1

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)

    if success:
        print(f"\n✅ Downloaded: {len(success)}")
        for symbol, tf in success:
            print(f"   ✅ {symbol} {tf}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for symbol, tf in failed:
            print(f"   ❌ {symbol} {tf}")

    print("\n" + "="*80)

    if failed:
        print("⚠️  Some downloads failed")
        return 1
    else:
        print("🎉 ALL DOWNLOADS COMPLETE!")
        print("\nNext step: Train models with:")
        print("  python train_all_models.py")
        return 0


if __name__ == '__main__':
    sys.exit(main())
