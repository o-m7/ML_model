#!/usr/bin/env python3
"""
DOWNLOAD OHLCV DATA FROM POLYGON S3 - SAVE AS PARQUET
======================================================

Downloads 5 years of OHLCV data for XAUUSD from Polygon S3 flat files.
This is the proper way to get historical data beyond what REST API offers.

Requires:
- POLYGON_S3_ACCESS_KEY
- POLYGON_S3_SECRET_KEY
- POLYGON_S3_ENDPOINT
- POLYGON_S3_BUCKET

Usage:
    python download_ohlcv_s3.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import io

import pandas as pd
import boto3
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

# S3 Credentials
S3_ACCESS_KEY = os.getenv('POLYGON_S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('POLYGON_S3_SECRET_KEY')
S3_ENDPOINT = os.getenv('POLYGON_S3_ENDPOINT', 'https://files.polygon.io')
S3_BUCKET = os.getenv('POLYGON_S3_BUCKET', 'flatfiles')

if not all([S3_ACCESS_KEY, S3_SECRET_KEY]):
    print("❌ ERROR: Missing S3 credentials in .env")
    print("   Required: POLYGON_S3_ACCESS_KEY, POLYGON_S3_SECRET_KEY")
    sys.exit(1)

# Configuration
SYMBOL = 'XAUUSD'
TICKER = 'C:XAUUSD'
TIMEFRAMES = {
    '1min': '1T',  # Polygon S3 uses 1min, we'll resample to our timeframes
}
DAYS_BACK = 1825  # 5 years

# Output directory
OUTPUT_DIR = Path("feature_store") / SYMBOL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print(f"DOWNLOADING OHLCV DATA FROM POLYGON S3 - {SYMBOL}")
print("="*80)


def get_s3_client():
    """Create S3 client for Polygon."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version='s3v4')
    )


def download_day_file(s3_client, date: datetime) -> pd.DataFrame:
    """
    Download OHLCV data for a specific day from S3.

    Polygon S3 structure:
    flatfiles/us_stocks_sip/minute_aggs_v1/{year}/{month:02d}/{ticker}/{year}-{month:02d}-{day:02d}.csv.gz
    """
    year = date.year
    month = date.month
    day = date.day

    # S3 key pattern for forex/crypto minute data
    # Adjust path based on your Polygon subscription
    key = f"forex/minute_aggs/{year}/{month:02d}/{TICKER}/{year}-{month:02d}-{day:02d}.csv.gz"

    try:
        print(f"   📥 Downloading {date.strftime('%Y-%m-%d')}...", end=' ')

        # Download file
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)

        # Read compressed CSV
        df = pd.read_csv(
            io.BytesIO(response['Body'].read()),
            compression='gzip'
        )

        if len(df) > 0:
            print(f"✅ {len(df):,} bars")
        else:
            print(f"⚠️  Empty")

        return df

    except s3_client.exceptions.NoSuchKey:
        print(f"⚠️  Not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()


def download_date_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Download data for a date range."""
    print(f"\n📡 Downloading from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    s3_client = get_s3_client()
    all_data = []

    current_date = start_date
    while current_date <= end_date:
        # Skip weekends (forex markets closed)
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            df_day = download_day_file(s3_client, current_date)
            if len(df_day) > 0:
                all_data.append(df_day)

        current_date += timedelta(days=1)

    if not all_data:
        print("\n   ❌ No data downloaded")
        return pd.DataFrame()

    # Concatenate all days
    print(f"\n   🔗 Combining {len(all_data)} days...")
    df = pd.concat(all_data, ignore_index=True)

    # Standardize columns
    # Polygon S3 CSV format: ticker,timestamp,open,high,low,close,volume,vwap,transactions
    df = df.rename(columns={
        'ticker': 'symbol',
        'timestamp': 'timestamp_ms',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vw': 'vwap',
        'n': 'transactions'
    })

    # Convert timestamp
    if 'timestamp_ms' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    elif 'timestamp' in df.columns and df['timestamp'].dtype == 'int64':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    # Sort and remove duplicates
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    print(f"   ✅ Total: {len(df):,} 1-minute bars")
    print(f"   📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1-minute data to higher timeframes."""
    print(f"\n🔄 Resampling to {timeframe}...")

    df = df.set_index('timestamp')

    # Resample
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })

    # Drop rows with NaN (weekends, gaps)
    resampled = resampled.dropna()

    # Reset index
    resampled = resampled.reset_index()

    print(f"   ✅ {len(resampled):,} {timeframe} bars")

    return resampled


def save_parquet(df: pd.DataFrame, timeframe: str):
    """Save DataFrame as parquet file."""
    output_path = OUTPUT_DIR / f"{SYMBOL}_{timeframe}.parquet"

    print(f"\n💾 Saving to {output_path}...")

    # Select final columns
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[columns]

    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved {len(df):,} rows ({file_size_mb:.2f} MB)")

    return output_path


def main():
    """Download and process OHLCV data."""

    # Date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS_BACK)

    print(f"\n📅 Fetching {DAYS_BACK} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

    # Download 1-minute data
    df_1min = download_date_range(start_date, end_date)

    if df_1min.empty:
        print("\n❌ Failed to download data")
        return 1

    # Resample to all timeframes
    results = []
    timeframes_to_create = {
        '5T': '5T',
        '15T': '15T',
        '30T': '30T',
        '1H': '1H',
        '4H': '4H'
    }

    for tf_name, tf_pandas in timeframes_to_create.items():
        print("\n" + "-"*80)

        # Resample
        df_resampled = resample_timeframe(df_1min.copy(), tf_pandas)

        # Save
        output_path = save_parquet(df_resampled, tf_name)

        results.append({
            'timeframe': tf_name,
            'bars': len(df_resampled),
            'file': output_path
        })

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)

    for result in results:
        print(f"✅ {result['timeframe']:6s}: {result['bars']:,} bars → {result['file'].name}")

    print("="*80)
    print(f"\n✅ Successfully created {len(results)} timeframes")
    print(f"📁 Files saved in: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Run: python download_quote_data.py  (optional - for quote features)")
    print("  2. Run: python calculate_all_features.py")
    print("  3. Run: python train_all_timeframes_local.py")
    print("\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
