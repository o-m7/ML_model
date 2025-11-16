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
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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

# Thread-safe printing
print_lock = Lock()


def fetch_quotes_day(date: datetime, ticker: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch quote data for a specific day and ticker from S3 using efficient download_file."""
    year = date.strftime('%Y')
    month = date.strftime('%m')
    date_str = date.strftime('%Y-%m-%d')

    object_key = f'global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz'

    # Use temp file for efficient download
    temp_file = None

    try:
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv.gz')
        temp_path = temp_file.name
        temp_file.close()

        # Download with retries
        for attempt in range(max_retries):
            try:
                s3.download_file(BUCKET, object_key, temp_path)
                break
            except s3.exceptions.NoSuchKey:
                return pd.DataFrame()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait_time)
                else:
                    if "NoSuchKey" not in str(e) and "404" not in str(e):
                        print(f"    ⚠️  Failed after {max_retries} retries on {date_str}: {e}")
                    return pd.DataFrame()

        # Read downloaded file
        with gzip.open(temp_path, 'rt') as gz:
            df = pd.read_csv(gz)

        rows_before_filter = len(df)

        # Filter for specific ticker
        if 'ticker' in df.columns:
            df = df[df['ticker'] == ticker].copy()

        rows_after_filter = len(df)

        if df.empty:
            return pd.DataFrame()

        # Debug: Show what columns we have on first day
        if date.day == 1 and date.month == 1:
            with print_lock:
                print(f"    📋 Debug {date_str}: {rows_before_filter:,} total rows, {rows_after_filter:,} for {ticker}")
                print(f"       Columns: {list(df.columns)[:10]}")

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

        rows_before_dropna = len(df)
        df = df.dropna()
        rows_after_dropna = len(df)

        # Debug on first day of year
        if date.day == 1 and date.month == 1 and rows_before_dropna > 0:
            with print_lock:
                dropped = rows_before_dropna - rows_after_dropna
                print(f"       Dropped {dropped:,} rows with NaN ({dropped/rows_before_dropna*100:.1f}%)")
                print(f"       Final: {rows_after_dropna:,} quotes for this day")

        return df

    except Exception as e:
        if "NoSuchKey" not in str(e) and "404" not in str(e):
            print(f"    ⚠️  Error on {date_str}: {e}")
        return pd.DataFrame()

    finally:
        # Clean up temp file
        if temp_file is not None:
            try:
                os.unlink(temp_path)
            except:
                pass


def download_quotes_month(symbol: str, year: int, month: int, month_idx: int = 0, total_months: int = 0) -> pd.DataFrame:
    """Download quotes for a single month, with caching. Thread-safe."""
    ticker = TICKER_MAP[symbol]

    # Check cache first
    cache_dir = Path("feature_store") / "quotes_cache" / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_{year}_{month:02d}_raw.parquet"

    if cache_file.exists():
        with print_lock:
            if total_months > 0:
                print(f"[{month_idx}/{total_months}] {year}-{month:02d} ✓ Using cache")
            else:
                print(f"  ✓ Using cached data for {year}-{month:02d}")
        return pd.read_parquet(cache_file)

    with print_lock:
        if total_months > 0:
            print(f"[{month_idx}/{total_months}] {year}-{month:02d} 📥 Downloading...")
        else:
            print(f"  📥 Downloading {year}-{month:02d}...")

    # Determine date range for this month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    all_data = []
    current_date = start_date
    days_processed = 0

    while current_date <= end_date:
        # Skip weekends (forex is closed)
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            df_day = fetch_quotes_day(current_date, ticker)

            if not df_day.empty:
                all_data.append(df_day)
                days_processed += 1

        current_date += timedelta(days=1)

    if not all_data:
        with print_lock:
            print(f"     ⚠️  No data found for {year}-{month:02d}")
        return pd.DataFrame()

    # Combine all days
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset='timestamp', keep='last')

    # Save to cache
    df.to_parquet(cache_file, index=False, compression='snappy')

    cache_size_mb = cache_file.stat().st_size / 1024 / 1024
    with print_lock:
        if total_months > 0:
            print(f"[{month_idx}/{total_months}] {year}-{month:02d} ✅ {len(df):,} quotes | {cache_size_mb:.1f} MB")
        else:
            print(f"     ✓ {len(df):,} quotes | {days_processed} days | {cache_size_mb:.1f} MB cached")

    return df


def download_quotes_range(symbol: str, start_date: str, end_date: str, max_workers: int = 3) -> pd.DataFrame:
    """Download quotes for date range, processing months in parallel with resume capability."""

    print(f"\n{'='*80}")
    print(f"DOWNLOADING {symbol} QUOTES FROM POLYGON S3")
    print(f"{'='*80}")
    print(f"Ticker: {TICKER_MAP[symbol]}")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")
    print(f"Strategy: Parallel monthly chunks ({max_workers} workers) with parquet caching")

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Generate list of (year, month) tuples to download
    months_to_download = []
    current = start
    while current <= end:
        months_to_download.append((current.year, current.month))
        # Move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    print(f"Total months: {len(months_to_download)}")
    print()

    # Download months in parallel
    monthly_data = {}  # Use dict to preserve order

    def download_wrapper(args):
        idx, year, month = args
        df = download_quotes_month(symbol, year, month, idx, len(months_to_download))
        return (year, month), df

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = []
        for idx, (year, month) in enumerate(months_to_download, 1):
            future = executor.submit(download_wrapper, (idx, year, month))
            futures.append(future)

        # Collect results as they complete
        for future in as_completed(futures):
            (year, month), df = future.result()
            if not df.empty:
                monthly_data[(year, month)] = df

    if not monthly_data:
        print(f"\n❌ No quote data found for {symbol}")
        return pd.DataFrame()

    # Sort by (year, month) and combine
    print(f"\n📊 Combining {len(monthly_data)} months of data...")
    sorted_months = sorted(monthly_data.keys())
    monthly_dfs = [monthly_data[key] for key in sorted_months]

    df = pd.concat(monthly_dfs, ignore_index=True)
    df = df.sort_values('timestamp')
    df = df.drop_duplicates(subset='timestamp', keep='last')

    print(f"\n✅ Downloaded {len(df):,} quote records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    if len(df) > 0:
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

    # Date range: 2022-2025 (recent 4 years for spread analysis)
    start_date = "2022-01-01"
    end_date = "2025-11-04"

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

    # Cache statistics
    print("\n📦 Cache Statistics:")
    cache_dir = Path("feature_store") / "quotes_cache"
    if cache_dir.exists():
        total_cache_mb = sum(f.stat().st_size for f in cache_dir.rglob("*.parquet")) / 1024 / 1024
        total_cache_files = len(list(cache_dir.rglob("*.parquet")))
        print(f"   Cached months: {total_cache_files}")
        print(f"   Cache size: {total_cache_mb:.1f} MB")
        print(f"   Location: {cache_dir}")
        print(f"\n   💡 To free space, delete: feature_store/quotes_cache/")
        print(f"      (Will re-download on next run)")
    else:
        print("   No cache files")


if __name__ == "__main__":
    # Check credentials
    if not os.getenv('Access_Key_ID'):
        print("❌ Missing Access_Key_ID in .env file")
        sys.exit(1)
    if not os.getenv('Secret_Access_Key'):
        print("❌ Missing Secret_Access_Key in .env file")
        sys.exit(1)

    main()
