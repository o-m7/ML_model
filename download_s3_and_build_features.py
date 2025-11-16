#!/usr/bin/env python3
"""
Download XAUUSD/XAGUSD from Polygon S3 (minute_aggs + quotes) and build all features.
Uses Massive.com S3 endpoint with proper configuration.
Date range: 2020-2025
"""

import os
import sys
import boto3
from botocore.config import Config
import pandas as pd
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# Initialize boto3 session with Massive.com configuration
session = boto3.Session(
    aws_access_key_id=os.getenv('Access_Key_ID'),
    aws_secret_access_key=os.getenv('Secret_Access_Key'),
)

s3 = session.client(
    's3',
    endpoint_url=os.getenv('S3_Endpoint', 'https://files.massive.com'),
    config=Config(signature_version='s3v4'),
)

BUCKET = os.getenv('Bucket', 'flatfiles')

# Symbols to download
SYMBOLS = ['XAUUSD', 'XAGUSD']
TICKER_MAP = {
    'XAUUSD': 'C:XAU-USD',
    'XAGUSD': 'C:XAG-USD',
}

# Timeframes to resample to
TIMEFRAMES = {
    '5T': 5,
    '15T': 15,
    '30T': 30,
    '1H': 60,
}


def fetch_minute_aggs(date: datetime, ticker: str) -> pd.DataFrame:
    """Fetch minute aggregates for a specific day and ticker from S3."""
    year = date.strftime('%Y')
    month = date.strftime('%m')
    date_str = date.strftime('%Y-%m-%d')

    object_key = f'global_forex/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'

    try:
        response = s3.get_object(Bucket=BUCKET, Key=object_key)
        compressed_data = response['Body'].read()

        with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
            df = pd.read_csv(gz)

        # Filter for specific ticker
        if 'ticker' in df.columns:
            df = df[df['ticker'] == ticker].copy()

        if df.empty:
            return pd.DataFrame()

        # Parse timestamp - Polygon uses nanoseconds in window_start
        if 'window_start' in df.columns:
            # Try nanoseconds first, then milliseconds, then as-is
            try:
                df['timestamp'] = pd.to_datetime(df['window_start'], unit='ns', utc=True)
            except:
                try:
                    df['timestamp'] = pd.to_datetime(df['window_start'], unit='ms', utc=True)
                except:
                    df['timestamp'] = pd.to_datetime(df['window_start'], utc=True)
        elif 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
            except:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                except:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Rename columns to standard OHLCV
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        })

        # Keep only required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]].copy()
        df = df.dropna()

        return df

    except s3.exceptions.NoSuchKey:
        return pd.DataFrame()
    except Exception as e:
        # Suppress common errors
        if "NoSuchKey" not in str(e) and "404" not in str(e):
            print(f"    ⚠️  Error on {date_str}: {e}")
        return pd.DataFrame()


def fetch_quotes(date: datetime, ticker: str) -> pd.DataFrame:
    """Fetch quote data for a specific day and ticker from S3."""
    year = date.strftime('%Y')
    month = date.strftime('%m')
    date_str = date.strftime('%Y-%m-%d')

    object_key = f'global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz'

    try:
        response = s3.get_object(Bucket=BUCKET, Key=object_key)
        compressed_data = response['Body'].read()

        with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
            df = pd.read_csv(gz)

        # Filter for specific ticker
        if 'ticker' in df.columns:
            df = df[df['ticker'] == ticker].copy()

        if df.empty:
            return pd.DataFrame()

        # Parse timestamp
        if 'participant_timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns', utc=True)
            except:
                try:
                    df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ms', utc=True)
                except:
                    df['timestamp'] = pd.to_datetime(df['participant_timestamp'], utc=True)
        elif 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
            except:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                except:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Keep relevant columns
        df = df.rename(columns={
            'bid_price': 'bid',
            'ask_price': 'ask',
            'bid_size': 'bid_size',
            'ask_size': 'ask_size',
        })

        # Calculate mid price for OHLCV
        if 'bid' in df.columns and 'ask' in df.columns:
            df['mid'] = (df['bid'] + df['ask']) / 2

        return df

    except s3.exceptions.NoSuchKey:
        return pd.DataFrame()
    except Exception as e:
        if "NoSuchKey" not in str(e) and "404" not in str(e):
            print(f"    ⚠️  Error on {date_str}: {e}")
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
    print(f"DOWNLOADING {symbol} FROM POLYGON S3")
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

            # Fetch minute aggregates (primary source)
            df_day = fetch_minute_aggs(current_date, ticker)

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

    # Date range: 2020-2025
    start_date = "2020-01-01"
    end_date = "2025-11-14"

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
    if not os.getenv('Access_Key_ID'):
        print("❌ Missing Access_Key_ID in .env file")
        sys.exit(1)
    if not os.getenv('Secret_Access_Key'):
        print("❌ Missing Secret_Access_Key in .env file")
        sys.exit(1)

    main()
