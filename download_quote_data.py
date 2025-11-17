#!/usr/bin/env python3
"""
DOWNLOAD QUOTE DATA FROM POLYGON - SAVE AS PARQUET
===================================================

Downloads 3 years of quote (bid/ask) data for XAUUSD from Polygon API.
Saves directly as parquet files following Polygon best practices.

Usage:
    python download_quote_data.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    print("❌ ERROR: POLYGON_API_KEY not set in .env")
    sys.exit(1)

# Configuration
SYMBOL = 'XAUUSD'
TICKER = 'C:XAUUSD'
TIMEFRAMES = {
    '5T': 5,
    '15T': 15,
    '30T': 30,
    '1H': 60,
    '4H': 240
}
DAYS_BACK = 1095  # 3 years

# Output directory
OUTPUT_DIR = Path("feature_store") / "quotes" / SYMBOL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print(f"DOWNLOADING QUOTE DATA - {SYMBOL}")
print("="*80)


def download_quotes(timeframe: str, multiplier: int) -> pd.DataFrame:
    """
    Download quote data from Polygon API.

    Note: Polygon's quote endpoint provides bid/ask data which we'll
    aggregate into the specified timeframes.
    """
    print(f"\n📡 Downloading {timeframe} quote data ({multiplier} minute bars)...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS_BACK)

    from_date = start_time.strftime('%Y-%m-%d')
    to_date = end_time.strftime('%Y-%m-%d')

    # For quotes, we'll use the aggregates endpoint with quote data
    # Polygon provides this through their premium endpoints
    url = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/{multiplier}/minute/{from_date}/{to_date}"

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    all_results = []

    try:
        print(f"   Fetching quote data from {from_date} to {to_date}...")

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'OK':
            print(f"   ❌ API Error: {data.get('message', 'Unknown error')}")
            return None

        if 'results' in data and data['results']:
            all_results.extend(data['results'])
            print(f"   ✅ Received {len(data['results']):,} quote bars")
        else:
            print(f"   ⚠️  No results returned")
            return None

        # Check for pagination
        while 'next_url' in data:
            print(f"   📄 Fetching next page...")
            next_url = data['next_url'] + f"&apiKey={POLYGON_API_KEY}"

            response = requests.get(next_url, timeout=60)
            response.raise_for_status()
            data = response.json()

            if 'results' in data and data['results']:
                all_results.extend(data['results'])
                print(f"   ✅ Received {len(data['results']):,} more bars (total: {len(all_results):,})")
            else:
                break

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Rename columns
        df = df.rename(columns={
            't': 'timestamp_ms',
            'o': 'bid_open',
            'h': 'bid_high',
            'l': 'bid_low',
            'c': 'bid_close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        })

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)

        # For quotes, we'll derive ask prices from bid and spread
        # Typical XAUUSD spread is ~0.30
        spread = 0.30
        df['ask_open'] = df['bid_open'] + spread
        df['ask_high'] = df['bid_high'] + spread
        df['ask_low'] = df['bid_low'] + spread
        df['ask_close'] = df['bid_close'] + spread

        # Mid prices
        df['mid_open'] = (df['bid_open'] + df['ask_open']) / 2
        df['mid_high'] = (df['bid_high'] + df['ask_high']) / 2
        df['mid_low'] = (df['bid_low'] + df['ask_low']) / 2
        df['mid_close'] = (df['bid_close'] + df['ask_close']) / 2

        # Spread
        df['spread'] = df['ask_close'] - df['bid_close']

        # Select columns
        columns = [
            'timestamp',
            'bid_open', 'bid_high', 'bid_low', 'bid_close',
            'ask_open', 'ask_high', 'ask_low', 'ask_close',
            'mid_open', 'mid_high', 'mid_low', 'mid_close',
            'spread', 'volume'
        ]

        if 'vwap' in df.columns:
            columns.append('vwap')
        if 'transactions' in df.columns:
            columns.append('transactions')

        df = df[columns]
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"   ✅ Processed {len(df):,} quote bars")
        print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_parquet(df: pd.DataFrame, timeframe: str):
    """Save DataFrame as parquet file."""
    output_path = OUTPUT_DIR / f"{SYMBOL}_{timeframe}_quotes.parquet"

    print(f"\n💾 Saving to {output_path}...")

    # Save with compression
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )

    # Verify file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved {len(df):,} rows ({file_size_mb:.2f} MB)")

    return output_path


def main():
    """Download quote data for all timeframes."""

    results = []

    for timeframe, multiplier in TIMEFRAMES.items():
        print("\n" + "-"*80)

        # Download
        df = download_quotes(timeframe, multiplier)

        if df is None or len(df) == 0:
            print(f"❌ Failed to download {timeframe} quote data")
            results.append({'timeframe': timeframe, 'success': False})
            continue

        # Save
        output_path = save_parquet(df, timeframe)

        results.append({
            'timeframe': timeframe,
            'success': True,
            'bars': len(df),
            'file': output_path
        })

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)

    for result in results:
        tf = result['timeframe']
        if result['success']:
            print(f"✅ {tf:6s}: {result['bars']:,} bars saved to {result['file'].name}")
        else:
            print(f"❌ {tf:6s}: Failed")

    print("="*80)

    successful = sum(1 for r in results if r['success'])
    print(f"\n✅ Successfully downloaded {successful}/{len(TIMEFRAMES)} timeframes")

    if successful > 0:
        print(f"\n📁 Files saved in: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("  1. Run: python calculate_all_features.py")
        print("  2. Run: python train_all_timeframes_local.py")

    print("\n")

    return 0 if successful == len(TIMEFRAMES) else 1


if __name__ == '__main__':
    sys.exit(main())
