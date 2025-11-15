#!/usr/bin/env python3
"""
DOWNLOAD AND PROCESS HISTORICAL FOREX DATA
===========================================
Downloads minute-level data from Massive.com S3 for 2019-2025
Processes XAUUSD and XAGUSD into multiple timeframes with features

Usage:
    python download_historical_data.py
    python download_historical_data.py --start-date 2019-01-01 --end-date 2025-11-14
    python download_historical_data.py --symbols XAUUSD XAGUSD --timeframes 5T 15T 30T 1H
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

# AWS credentials (from .env or hardcoded)
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', '4937f95b-db8b-4d7e-8d54-756a82d4976e')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1')
ENDPOINT_URL = 'https://files.massive.com'
BUCKET_NAME = 'flatfiles'

# Symbols to download
SYMBOLS = ['XAUUSD', 'XAGUSD']
TIMEFRAMES = ['5T', '15T', '30T', '1H']

# Paths
RAW_DATA_DIR = Path('raw_data_cache')
FEATURE_STORE_DIR = Path('feature_store')


def setup_s3_client():
    """Initialize S3 client for Massive.com."""
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    s3 = session.client(
        's3',
        endpoint_url=ENDPOINT_URL,
        config=Config(signature_version='s3v4'),
    )

    return s3


def download_daily_file(s3, date_str, force=False):
    """
    Download a single day's data file from S3.

    Args:
        s3: boto3 S3 client
        date_str: Date string in YYYY-MM-DD format
        force: Force re-download even if file exists

    Returns:
        Path to downloaded file, or None if failed
    """
    # Parse date
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')

    # Build S3 object key
    object_key = f'flatfiles/global_forex/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'

    # Remove bucket prefix if present
    if object_key.startswith(BUCKET_NAME + '/'):
        object_key = object_key[len(BUCKET_NAME + '/'):]

    # Local file path
    local_file = RAW_DATA_DIR / f"{date_str}.csv.gz"

    # Skip if already exists
    if local_file.exists() and not force:
        print(f"  ✓ Already downloaded: {date_str}")
        return local_file

    # Download
    try:
        print(f"  ⬇️  Downloading: {date_str}...", end=' ', flush=True)
        s3.download_file(BUCKET_NAME, object_key, str(local_file))
        print("✓")
        return local_file

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def read_and_filter_gz(file_path, symbols):
    """
    Read gzipped CSV and filter for specific symbols.

    Args:
        file_path: Path to .csv.gz file
        symbols: List of symbols to filter (e.g., ['XAUUSD', 'XAGUSD'])

    Returns:
        DataFrame with filtered data
    """
    try:
        with gzip.open(file_path, 'rt') as f:
            # Read CSV
            df = pd.read_csv(f)

        # Expected columns: ticker, timestamp, open, high, low, close, volume
        # Filter for our symbols
        if 'ticker' in df.columns:
            df = df[df['ticker'].isin(symbols)].copy()
        elif 'symbol' in df.columns:
            df = df[df['symbol'].isin(symbols)].copy()
        else:
            print(f"  ⚠️  Warning: No ticker/symbol column in {file_path}")
            return pd.DataFrame()

        if df.empty:
            return df

        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        elif 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'], utc=True)

        # Standardize column names
        df = df.rename(columns={
            'ticker': 'symbol',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            't': 'timestamp'
        })

        # Keep only needed columns
        cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]

        return df

    except Exception as e:
        print(f"  ✗ Error reading {file_path}: {e}")
        return pd.DataFrame()


def resample_ohlcv(df, timeframe):
    """
    Resample OHLCV data to a specific timeframe.

    Args:
        df: DataFrame with timestamp index and OHLCV columns
        timeframe: Pandas resample string (e.g., '5T', '15T', '1H')

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Resample
    df_resampled = df.resample(timeframe, label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return df_resampled


def download_date_range(start_date, end_date, symbols=SYMBOLS):
    """
    Download all daily files in a date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbols: List of symbols to download

    Returns:
        Dict mapping symbol -> DataFrame with all data
    """
    # Setup
    RAW_DATA_DIR.mkdir(exist_ok=True)
    s3 = setup_s3_client()

    # Date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    current = start
    all_data = {symbol: [] for symbol in symbols}

    print(f"\n{'='*80}")
    print(f"DOWNLOADING DATA: {start_date} to {end_date}")
    print(f"Symbols: {symbols}")
    print(f"{'='*80}\n")

    downloaded_count = 0
    error_count = 0

    # Download each day
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        file_path = download_daily_file(s3, date_str)

        if file_path and file_path.exists():
            # Read and filter
            df = read_and_filter_gz(file_path, symbols)

            if not df.empty:
                # Split by symbol
                for symbol in symbols:
                    symbol_data = df[df['symbol'] == symbol].copy()
                    if not symbol_data.empty:
                        all_data[symbol].append(symbol_data)

            downloaded_count += 1
        else:
            error_count += 1

        # Move to next day
        current += timedelta(days=1)

    print(f"\n✅ Downloaded: {downloaded_count} days")
    if error_count > 0:
        print(f"⚠️  Errors: {error_count} days")

    # Combine all data per symbol
    combined_data = {}
    for symbol in symbols:
        if all_data[symbol]:
            print(f"\n📊 Combining data for {symbol}...")
            df_combined = pd.concat(all_data[symbol], ignore_index=True)

            # Sort by timestamp
            df_combined = df_combined.sort_values('timestamp')

            # Remove duplicates
            df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='first')

            # Set index
            df_combined = df_combined.set_index('timestamp')

            # Drop symbol column
            if 'symbol' in df_combined.columns:
                df_combined = df_combined.drop(columns=['symbol'])

            combined_data[symbol] = df_combined

            print(f"  ✅ {symbol}: {len(df_combined):,} bars from {df_combined.index[0]} to {df_combined.index[-1]}")
        else:
            print(f"  ⚠️  {symbol}: No data")
            combined_data[symbol] = pd.DataFrame()

    return combined_data


def process_and_save(symbol_data, timeframes=TIMEFRAMES):
    """
    Resample to multiple timeframes and calculate features.

    Args:
        symbol_data: Dict mapping symbol -> raw minute DataFrame
        timeframes: List of timeframes to create

    Returns:
        Dict mapping (symbol, timeframe) -> feature DataFrame
    """
    results = {}

    for symbol, df_raw in symbol_data.items():
        if df_raw.empty:
            print(f"\n⚠️  Skipping {symbol}: No data")
            continue

        print(f"\n{'='*80}")
        print(f"PROCESSING: {symbol}")
        print(f"{'='*80}")

        symbol_dir = FEATURE_STORE_DIR / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        for timeframe in timeframes:
            print(f"\n🔧 {symbol} {timeframe}:")

            # Resample
            print(f"  Resampling to {timeframe}...", end=' ', flush=True)
            df_resampled = resample_ohlcv(df_raw.copy(), timeframe)
            print(f"✓ {len(df_resampled):,} bars")

            if df_resampled.empty:
                print(f"  ⚠️  No data after resampling")
                continue

            # Calculate features
            print(f"  Calculating features...", end=' ', flush=True)
            df_features = build_feature_frame(df_resampled)
            print(f"✓ {len(df_features.columns)} columns")

            # Save to parquet
            output_path = symbol_dir / f"{symbol}_{timeframe}.parquet"
            print(f"  Saving to {output_path}...", end=' ', flush=True)
            df_features.to_parquet(output_path)
            print(f"✓")

            results[(symbol, timeframe)] = df_features

            print(f"  ✅ {symbol} {timeframe}: {len(df_features):,} bars with features")

    return results


def main():
    parser = argparse.ArgumentParser(description='Download and process historical forex data')
    parser.add_argument('--start-date', type=str, default='2019-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-11-14',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS,
                       help='Symbols to download')
    parser.add_argument('--timeframes', nargs='+', default=TIMEFRAMES,
                       help='Timeframes to create')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download, only process existing raw data')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("HISTORICAL DATA DOWNLOAD AND FEATURE EXTRACTION")
    print("="*80)
    print(f"\nDate range: {args.start_date} to {args.end_date}")
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")

    # Download data
    if not args.skip_download:
        symbol_data = download_date_range(args.start_date, args.end_date, args.symbols)
    else:
        print("\n⚠️  Skipping download (--skip-download)")
        # Load from cache if available
        symbol_data = {}
        for symbol in args.symbols:
            cache_file = RAW_DATA_DIR / f"{symbol}_combined.parquet"
            if cache_file.exists():
                symbol_data[symbol] = pd.read_parquet(cache_file)
                print(f"  ✅ Loaded cached {symbol}: {len(symbol_data[symbol]):,} bars")
            else:
                symbol_data[symbol] = pd.DataFrame()
                print(f"  ⚠️  No cached data for {symbol}")

    # Save combined raw data to cache
    if not args.skip_download:
        print(f"\n💾 Saving combined raw data to cache...")
        for symbol, df in symbol_data.items():
            if not df.empty:
                cache_file = RAW_DATA_DIR / f"{symbol}_combined.parquet"
                df.to_parquet(cache_file)
                print(f"  ✅ {symbol}: {cache_file}")

    # Process and save
    results = process_and_save(symbol_data, args.timeframes)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for (symbol, timeframe), df in results.items():
        days = (df.index[-1] - df.index[0]).days if len(df) > 0 else 0
        print(f"{symbol} {timeframe}: {len(df):,} bars | {days} days | {len(df.columns)} features")

    print(f"\n✅ Data saved to: {FEATURE_STORE_DIR}/")
    print(f"✅ Cache saved to: {RAW_DATA_DIR}/")

    print(f"\nNext steps:")
    print(f"  1. Run training: python citadel_training_system.py")
    print(f"  2. Or use existing: python retrain_all_temporal.py")


if __name__ == '__main__':
    main()
