#!/usr/bin/env python3
"""
Download XAUUSD/XAGUSD from Polygon S3 minute aggregates (2019-2025) and build all features.
Uses direct S3 access to Polygon flat files.
"""

import os
import sys
import boto3
import pandas as pd
import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# S3 Configuration
S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=os.getenv('POLYGON_S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('POLYGON_S3_SECRET_KEY'),
    endpoint_url=os.getenv('POLYGON_S3_ENDPOINT')
)

S3_BUCKET = os.getenv('POLYGON_S3_BUCKET', 'flatfiles')

# Symbols to download
SYMBOLS = ['XAUUSD', 'XAGUSD']
TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
}

# Timeframes to resample to
TIMEFRAMES = {
    '5T': 5,
    '15T': 15,
    '30T': 30,
    '1H': 60,
}


def fetch_day_from_s3(date: datetime, ticker: str) -> pd.DataFrame:
    """Fetch minute aggregates for a specific day and ticker from S3."""
    year = date.strftime('%Y')
    month = date.strftime('%m')
    date_str = date.strftime('%Y-%m-%d')

    file_key = f'global_forex/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'

    try:
        # Fetch file from S3
        response = S3_CLIENT.get_object(Bucket=S3_BUCKET, Key=file_key)
        compressed_data = response['Body'].read()

        # Decompress and read CSV
        with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
            df = pd.read_csv(gz)

        # Filter for specific ticker
        if 'ticker' in df.columns:
            df = df[df['ticker'] == ticker].copy()

        if df.empty:
            return pd.DataFrame()

        # Convert timestamp (Polygon uses milliseconds)
        if 'window_start' in df.columns:
            df['timestamp'] = pd.to_datetime(df['window_start'], unit='ms', utc=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Rename columns to standard OHLCV
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'transactions': 'volume'  # Fallback
        })

        # Keep only required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna()

        return df

    except S3_CLIENT.exceptions.NoSuchKey:
        return pd.DataFrame()  # File doesn't exist
    except Exception as e:
        print(f"    ⚠️  Error: {e}")
        return pd.DataFrame()


def download_symbol_range(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download minute aggregates for a symbol from S3 for date range.

    Args:
        symbol: 'XAUUSD' or 'XAGUSD'
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'

    Returns:
        DataFrame with minute bars
    """
    ticker = TICKER_MAP[symbol]

    print(f"\n{'='*80}")
    print(f"DOWNLOADING {symbol} MINUTE AGGREGATES FROM S3")
    print(f"{'='*80}")
    print(f"Ticker: {ticker}")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    all_data = []
    current_date = start
    total_days = (end - start).days
    day_count = 0

    while current_date <= end:
        # Skip weekends (forex is closed)
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            day_count += 1

            if day_count % 10 == 0:
                progress = (current_date - start).days / total_days * 100
                print(f"  Progress: {progress:.1f}% ({current_date.strftime('%Y-%m-%d')})")

            df_day = fetch_day_from_s3(current_date, ticker)

            if not df_day.empty:
                all_data.append(df_day)

        current_date += timedelta(days=1)

    if not all_data:
        print(f"\n❌ No data found for {symbol}")
        return pd.DataFrame()

    # Combine all days
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset='timestamp', keep='last')

    print(f"\n✅ Downloaded {len(df):,} minute bars")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")

    return df


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample minute data to specified timeframe."""
    if df.empty:
        return df

    df = df.set_index('timestamp').sort_index()

    # Resample
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    resampled = resampled.reset_index()

    return resampled


def save_to_feature_store(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save data to feature store."""
    output_dir = Path("feature_store") / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{symbol}_{timeframe}.parquet"

    df.to_parquet(output_file, index=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024

    print(f"\n💾 Saved to {output_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Size: {file_size_mb:.2f} MB")


def main():
    """Main download and processing pipeline."""

    # Date range
    start_date = "2019-01-01"
    end_date = "2024-11-14"

    print("\n" + "="*80)
    print("POLYGON S3 DATA DOWNLOAD & FEATURE GENERATION")
    print("="*80)
    print(f"\nDate range: {start_date} to {end_date}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES.keys())}")

    for symbol in SYMBOLS:
        print(f"\n{'#'*80}")
        print(f"# PROCESSING {symbol}")
        print(f"{'#'*80}")

        # Download minute data
        df_minute = download_symbol_range(symbol, start_date, end_date)

        if df_minute.empty:
            print(f"❌ Skipping {symbol} - no data")
            continue

        # Resample to all timeframes
        for timeframe, minutes in TIMEFRAMES.items():
            print(f"\n📊 Resampling to {timeframe} ({minutes} minutes)...")

            df_resampled = resample_to_timeframe(df_minute.copy(), timeframe)

            if not df_resampled.empty:
                print(f"   ✅ {len(df_resampled):,} bars")

                # Save to feature store
                save_to_feature_store(df_resampled, symbol, timeframe)
            else:
                print(f"   ⚠️  No data after resampling")

    print("\n" + "="*80)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*80)

    # Summary
    print("\n📊 Summary:")
    feature_store = Path("feature_store")
    for symbol in SYMBOLS:
        print(f"\n{symbol}:")
        symbol_dir = feature_store / symbol
        if symbol_dir.exists():
            for timeframe in TIMEFRAMES.keys():
                file_path = symbol_dir / f"{symbol}_{timeframe}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
                    print(f"  {timeframe:5s}: {len(df):7,} bars | {size_mb:6.2f} MB | {date_range:4d} days")

    print("\n🚀 Ready to run institutional ML system!")
    print("   Run: python3 institutional_ml_trading_system.py")


if __name__ == "__main__":
    # Check credentials
    if not os.getenv('POLYGON_S3_ACCESS_KEY'):
        print("❌ Missing POLYGON_S3_ACCESS_KEY in .env file")
        sys.exit(1)

    main()
