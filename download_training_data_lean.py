#!/usr/bin/env python3
"""
SPACE-EFFICIENT HISTORICAL DATA DOWNLOAD
=========================================
Downloads and processes data incrementally to minimize disk usage.
Deletes raw files after processing to save space.

Usage:
    python download_training_data_lean.py
    python download_training_data_lean.py --start-date 2020-01-01  # Last 5 years only
"""

import argparse
import gzip
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

# Import feature calculation
from live_feature_utils import build_feature_frame

# Load environment variables
load_dotenv()

# AWS credentials
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', '4937f95b-db8b-4d7e-8d54-756a82d4976e')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1')
ENDPOINT_URL = 'https://files.massive.com'
BUCKET_NAME = 'flatfiles'

# Symbols
SYMBOLS = ['XAUUSD', 'XAGUSD']
TIMEFRAMES = ['5T', '15T', '30T', '1H']

# Paths
TEMP_DIR = Path('temp_download')  # Temporary storage
FEATURE_STORE_DIR = Path('feature_store')


def setup_s3_client():
    """Initialize S3 client."""
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    return session.client(
        's3',
        endpoint_url=ENDPOINT_URL,
        config=Config(signature_version='s3v4'),
    )


def download_and_parse_day(s3, date_str, symbols):
    """
    Download a single day, parse it, and return data.
    Deletes the file after parsing to save space.

    Returns:
        Dict mapping symbol -> DataFrame for this day
    """
    # Parse date
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')

    # Build S3 object key
    object_key = f'flatfiles/global_forex/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'
    if object_key.startswith(BUCKET_NAME + '/'):
        object_key = object_key[len(BUCKET_NAME + '/'):]

    # Temporary file
    temp_file = TEMP_DIR / f"{date_str}.csv.gz"

    # Download
    try:
        s3.download_file(BUCKET_NAME, object_key, str(temp_file))

        # Parse immediately
        with gzip.open(temp_file, 'rt') as f:
            df = pd.read_csv(f)

        # Delete downloaded file to save space
        temp_file.unlink()

        # Filter and standardize
        if 'ticker' in df.columns:
            df = df[df['ticker'].isin(symbols)].copy()
            df = df.rename(columns={'ticker': 'symbol'})
        elif 'symbol' in df.columns:
            df = df[df['symbol'].isin(symbols)].copy()
        else:
            return {}

        if df.empty:
            return {}

        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        elif 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'], utc=True)
            df = df.rename(columns={'t': 'timestamp'})

        # Standardize column names
        df = df.rename(columns={
            'o': 'open', 'h': 'high', 'l': 'low',
            'c': 'close', 'v': 'volume'
        })

        # Keep only needed columns
        cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]

        # Split by symbol
        result = {}
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            if not symbol_data.empty:
                symbol_data = symbol_data.drop(columns=['symbol'])
                symbol_data = symbol_data.set_index('timestamp')
                result[symbol] = symbol_data

        return result

    except Exception as e:
        # Clean up on error
        if temp_file.exists():
            temp_file.unlink()
        return {}


def download_and_combine(start_date, end_date, symbols):
    """
    Download all data and combine incrementally.
    Saves to checkpoint files every 100 days to avoid data loss.

    Returns:
        Dict mapping symbol -> combined DataFrame
    """
    TEMP_DIR.mkdir(exist_ok=True)
    s3 = setup_s3_client()

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Initialize data collectors
    all_data = {symbol: [] for symbol in symbols}

    print(f"\n{'='*80}")
    print(f"DOWNLOADING: {start_date} to {end_date}")
    print(f"Symbols: {symbols}")
    print(f"{'='*80}\n")

    current = start
    day_count = 0
    success_count = 0

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        # Download and parse this day
        day_data = download_and_parse_day(s3, date_str, symbols)

        if day_data:
            for symbol in symbols:
                if symbol in day_data:
                    all_data[symbol].append(day_data[symbol])
            success_count += 1
            print(f"✓ {date_str}", end='\r', flush=True)
        else:
            print(f"⚠ {date_str}", end='\r', flush=True)

        day_count += 1

        # Save checkpoint every 100 days
        if day_count % 100 == 0:
            print(f"\n📊 Checkpoint: {day_count} days processed, {success_count} successful")

            # Save intermediate checkpoint
            for symbol in symbols:
                if all_data[symbol]:
                    checkpoint_file = TEMP_DIR / f"{symbol}_checkpoint.parquet"
                    df_temp = pd.concat(all_data[symbol], axis=0)
                    df_temp = df_temp.sort_index().drop_duplicates()
                    df_temp.to_parquet(checkpoint_file)

        current += timedelta(days=1)

    print(f"\n\n✅ Downloaded: {success_count}/{day_count} days")

    # Combine all data
    combined = {}
    for symbol in symbols:
        if all_data[symbol]:
            print(f"\n📊 Combining {symbol}...")
            df = pd.concat(all_data[symbol], axis=0)
            df = df.sort_index().drop_duplicates()
            combined[symbol] = df
            print(f"  ✅ {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
        else:
            combined[symbol] = pd.DataFrame()

    # Clean up temp directory
    for f in TEMP_DIR.glob('*'):
        f.unlink()

    return combined


def resample_ohlcv(df, timeframe):
    """Resample OHLCV data."""
    if df.empty:
        return df

    return df.resample(timeframe, label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


def process_and_save(symbol_data, timeframes):
    """
    Process each symbol and save to feature_store.
    Does NOT keep intermediate files to save space.
    """
    for symbol, df_raw in symbol_data.items():
        if df_raw.empty:
            print(f"\n⚠️  No data for {symbol}")
            continue

        print(f"\n{'='*80}")
        print(f"PROCESSING: {symbol}")
        print(f"{'='*80}")

        symbol_dir = FEATURE_STORE_DIR / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        for timeframe in timeframes:
            print(f"\n🔧 {timeframe}:", end=' ')

            # Resample
            df_resampled = resample_ohlcv(df_raw.copy(), timeframe)
            print(f"{len(df_resampled):,} bars", end=' → ')

            if df_resampled.empty:
                print("❌ No data")
                continue

            # Calculate features
            df_features = build_feature_frame(df_resampled)
            print(f"{len(df_features.columns)} features", end=' → ')

            # Save
            output_path = symbol_dir / f"{symbol}_{timeframe}.parquet"
            df_features.to_parquet(output_path)

            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Download forex data (space-efficient)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date (default: 2020-01-01 for last 5 years)')
    parser.add_argument('--end-date', type=str, default='2025-11-14',
                       help='End date')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS,
                       help='Symbols to download')
    parser.add_argument('--timeframes', nargs='+', default=TIMEFRAMES,
                       help='Timeframes to create')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SPACE-EFFICIENT DATA DOWNLOAD")
    print("="*80)
    print(f"\nPeriod: {args.start_date} to {args.end_date}")
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"\n⚠️  Raw files deleted after processing to save space")

    # Download and combine
    symbol_data = download_and_combine(args.start_date, args.end_date, args.symbols)

    # Process and save
    process_and_save(symbol_data, args.timeframes)

    # Summary
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")

    for symbol in args.symbols:
        for timeframe in args.timeframes:
            file_path = FEATURE_STORE_DIR / symbol / f"{symbol}_{timeframe}.parquet"
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                df = pd.read_parquet(file_path)
                days = (df.index[-1] - df.index[0]).days
                print(f"{symbol} {timeframe}: {len(df):,} bars | {days} days | {size_mb:.1f} MB")

    print(f"\n✅ Data saved to: {FEATURE_STORE_DIR}/")
    print(f"\nNext: python citadel_training_system.py")


if __name__ == '__main__':
    main()
