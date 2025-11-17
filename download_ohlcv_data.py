#!/usr/bin/env python3
"""
DOWNLOAD OHLCV DATA FROM POLYGON - SAVE AS PARQUET
===================================================

Downloads 5 years of OHLCV data for XAUUSD from Polygon API.
Saves directly as parquet files following Polygon best practices.

Usage:
    python download_ohlcv_data.py
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
DAYS_BACK = 1825  # 5 years

# Output directory
OUTPUT_DIR = Path("feature_store") / SYMBOL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print(f"DOWNLOADING OHLCV DATA - {SYMBOL}")
print("="*80)


def download_ohlcv(timeframe: str, multiplier: int) -> pd.DataFrame:
    """
    Download OHLCV data from Polygon API.

    Following Polygon's recommended approach:
    - Use /v2/aggs/ticker endpoint
    - Request large date ranges
    - Handle pagination properly
    """
    print(f"\n📡 Downloading {timeframe} data ({multiplier} minute bars)...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS_BACK)

    from_date = start_time.strftime('%Y-%m-%d')
    to_date = end_time.strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/{multiplier}/minute/{from_date}/{to_date}"

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,  # Max results per request
        'apiKey': POLYGON_API_KEY
    }

    all_results = []

    try:
        print(f"   Fetching data from {from_date} to {to_date}...")

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'OK':
            print(f"   ❌ API Error: {data.get('message', 'Unknown error')}")
            return None

        if 'results' in data and data['results']:
            all_results.extend(data['results'])
            print(f"   ✅ Received {len(data['results']):,} bars")
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

        # Rename columns to standard OHLCV format
        df = df.rename(columns={
            't': 'timestamp_ms',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        })

        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)

        # Select and order columns
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if 'vwap' in df.columns:
            columns.append('vwap')
        if 'transactions' in df.columns:
            columns.append('transactions')

        df = df[columns]
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"   ✅ Processed {len(df):,} bars")
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
    output_path = OUTPUT_DIR / f"{SYMBOL}_{timeframe}.parquet"

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
    """Download OHLCV data for all timeframes."""

    results = []

    for timeframe, multiplier in TIMEFRAMES.items():
        print("\n" + "-"*80)

        # Download
        df = download_ohlcv(timeframe, multiplier)

        if df is None or len(df) == 0:
            print(f"❌ Failed to download {timeframe} data")
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
        print("  1. Run: python download_quote_data.py")
        print("  2. Run: python calculate_all_features.py")
        print("  3. Run: python train_all_timeframes_local.py")

    print("\n")

    return 0 if successful == len(TIMEFRAMES) else 1


if __name__ == '__main__':
    sys.exit(main())
