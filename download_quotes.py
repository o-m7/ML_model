#!/usr/bin/env python3
"""
Download quote data (bid/ask spreads) from Polygon S3.
Saves to feature_store/quotes/ directory.
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


def fetch_quotes_day(date: datetime, ticker: str) -> pd.DataFrame:
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

        # Parse timestamp - try multiple formats
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

        # Rename columns
        df = df.rename(columns={
            'bid_price': 'bid',
            'ask_price': 'ask',
            'bid_size': 'bid_size',
            'ask_size': 'ask_size',
        })

        # Calculate spread and mid price
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread'] = df['ask'] - df['bid']
            df['spread_pct'] = (df['spread'] / df['bid']) * 100
            df['mid'] = (df['bid'] + df['ask']) / 2

        # Keep relevant columns
        keep_cols = ['timestamp', 'bid', 'ask', 'spread', 'spread_pct', 'mid']
        if 'bid_size' in df.columns:
            keep_cols.extend(['bid_size', 'ask_size'])

        df = df[[col for col in keep_cols if col in df.columns]].copy()
        df = df.dropna()

        return df

    except s3.exceptions.NoSuchKey:
        return pd.DataFrame()
    except Exception as e:
        if "NoSuchKey" not in str(e) and "404" not in str(e):
            print(f"    ⚠️  Error on {date_str}: {e}")
        return pd.DataFrame()


def download_quotes_range(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download quotes for date range."""
    ticker = TICKER_MAP[symbol]

    print(f"\n{'='*80}")
    print(f"DOWNLOADING {symbol} QUOTES FROM POLYGON S3")
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

            df_day = fetch_quotes_day(current_date, ticker)

            if not df_day.empty:
                all_data.append(df_day)

        current_date += timedelta(days=1)

    if not all_data:
        print(f"\n❌ No quote data found for {symbol}")
        return pd.DataFrame()

    # Combine all days
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset='timestamp', keep='last')

    print(f"\n✅ Downloaded {len(df):,} quote records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Avg spread: {df['spread'].mean():.4f}")
    print(f"   Avg spread %: {df['spread_pct'].mean():.4f}%")

    return df


def resample_quotes(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample quote data to specified timeframe."""
    if df.empty:
        return df

    df = df.set_index('timestamp').sort_index()

    # Resample quotes - use mean for spread metrics
    agg_dict = {
        'bid': 'mean',
        'ask': 'mean',
        'spread': 'mean',
        'spread_pct': 'mean',
        'mid': 'mean',
    }

    if 'bid_size' in df.columns:
        agg_dict['bid_size'] = 'sum'
        agg_dict['ask_size'] = 'sum'

    resampled = df.resample(timeframe).agg(agg_dict).dropna()
    resampled = resampled.reset_index()

    return resampled


def save_quotes(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save quote data to feature store."""
    output_dir = Path("feature_store") / "quotes" / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{symbol}_{timeframe}_quotes.parquet"

    df.to_parquet(output_file, index=False, compression='snappy')

    file_size_mb = output_file.stat().st_size / 1024 / 1024

    print(f"\n💾 Saved to {output_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Size: {file_size_mb:.2f} MB")


def main():
    """Main download pipeline for quotes."""

    # Date range: 2020-2025
    start_date = "2020-01-01"
    end_date = "2025-11-14"

    TIMEFRAMES = ['5T', '15T', '30T', '1H']

    print("\n" + "="*80)
    print("POLYGON S3 QUOTE DATA DOWNLOAD")
    print("="*80)
    print(f"\nDate range: {start_date} to {end_date}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")

    for symbol in SYMBOLS:
        print(f"\n{'#'*80}")
        print(f"# PROCESSING {symbol} QUOTES")
        print(f"{'#'*80}")

        # Download raw quote data
        df_quotes = download_quotes_range(symbol, start_date, end_date)

        if df_quotes.empty:
            print(f"❌ Skipping {symbol} - no quote data")
            continue

        # Resample to all timeframes
        for timeframe in TIMEFRAMES:
            print(f"\n📊 Resampling quotes to {timeframe}...")

            df_resampled = resample_quotes(df_quotes.copy(), timeframe)

            if not df_resampled.empty:
                print(f"   ✅ {len(df_resampled):,} bars")
                save_quotes(df_resampled, symbol, timeframe)
            else:
                print(f"   ⚠️  No data after resampling")

    print("\n" + "="*80)
    print("✅ QUOTE DOWNLOAD COMPLETE!")
    print("="*80)

    # Summary
    print("\n📊 Summary:")
    quotes_dir = Path("feature_store") / "quotes"

    for symbol in SYMBOLS:
        print(f"\n{symbol}:")
        symbol_dir = quotes_dir / symbol

        if symbol_dir.exists():
            for timeframe in TIMEFRAMES:
                file_path = symbol_dir / f"{symbol}_{timeframe}_quotes.parquet"

                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    avg_spread = df['spread_pct'].mean()
                    print(f"  {timeframe:5s}: {len(df):7,} bars | Avg spread: {avg_spread:.4f}% | {size_mb:6.2f} MB")


if __name__ == "__main__":
    # Check credentials
    if not os.getenv('Access_Key_ID'):
        print("❌ Missing Access_Key_ID in .env file")
        sys.exit(1)
    if not os.getenv('Secret_Access_Key'):
        print("❌ Missing Secret_Access_Key in .env file")
        sys.exit(1)

    main()
