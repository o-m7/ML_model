#!/usr/bin/env python3
"""
Download historical data (2019-2025) and build feature parquet files.
Uses the same working approach as the original 54-day dataset.

Process:
1. Download raw OHLCV from Polygon S3
2. Calculate features using live_feature_utils
3. Save to parquet
4. Delete raw data to save space
"""

import os
import sys
import gzip
import boto3
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

# Import feature calculation
from live_feature_utils import build_feature_frame

load_dotenv()

# S3 Configuration (Polygon S3)
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', '4937f95b-db8b-4d7e-8d54-756a82d4976e')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1')
ENDPOINT_URL = os.getenv('POLYGON_S3_ENDPOINT', 'https://files.massive.com')
S3_BUCKET = os.getenv('POLYGON_S3_BUCKET', 'flatfiles')

# Symbols to download
SYMBOLS = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
}

TIMEFRAMES = ['5T', '15T', '30T', '1H']

# Paths
FEATURE_STORE_DIR = Path('feature_store')


def setup_s3_client():
    """Initialize S3 client."""
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    return session.client('s3', endpoint_url=ENDPOINT_URL)


def download_day(s3, date_str, polygon_ticker):
    """
    Download one day of minute data from S3.

    Returns:
        DataFrame with OHLCV minute data, or None if not found
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    # Skip weekends
    if date_obj.weekday() >= 5:
        return None

    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')

    # S3 key (NO 'flatfiles/' prefix - that's the bucket name)
    file_key = f'global_forex/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'

    try:
        # Fetch from S3
        response = s3.get_object(Bucket=S3_BUCKET, Key=file_key)
        compressed_data = response['Body'].read()

        # Decompress and parse
        with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
            df = pd.read_csv(gz)

        # Filter for our ticker
        if 'ticker' in df.columns:
            df = df[df['ticker'] == polygon_ticker].copy()
        else:
            return None

        if df.empty:
            return None

        # Convert timestamp (Polygon uses milliseconds)
        if 'window_start' in df.columns:
            df['timestamp'] = pd.to_datetime(df['window_start'], unit='ms', utc=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            return None

        # Standardize columns
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'transactions': 'volume'  # Fallback
        })

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.set_index('timestamp').sort_index()

        return df

    except s3.exceptions.NoSuchKey:
        return None  # No data for this date
    except Exception as e:
        print(f"    ⚠️  Error on {date_str}: {e}")
        return None


def download_and_process_symbol(symbol, start_date, end_date):
    """
    Download all data for one symbol and create feature parquet files.

    Process:
    1. Download minute bars for date range
    2. Resample to each timeframe
    3. Calculate features
    4. Save to parquet
    """
    polygon_ticker = SYMBOLS[symbol]

    print(f"\n{'='*80}")
    print(f"Processing {symbol} ({polygon_ticker})")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*80}\n")

    s3 = setup_s3_client()

    # Calculate total days
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - current).days + 1

    # Download all minute data
    print("📥 Downloading minute data...")
    print(f"   Total days to process: {total_days}")
    print()

    all_minute_bars = []
    day_count = 0
    success_count = 0

    # Progress tracking
    last_update = datetime.now()
    start_time = datetime.now()

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        df_day = download_day(s3, date_str, polygon_ticker)
        if df_day is not None:
            all_minute_bars.append(df_day)
            success_count += 1

        day_count += 1

        # Update progress every second or every 50 days
        now = datetime.now()
        if (now - last_update).total_seconds() >= 1.0 or day_count % 50 == 0:
            progress_pct = (day_count / total_days) * 100
            elapsed = (now - start_time).total_seconds()

            # Estimate time remaining
            if day_count > 0:
                rate = day_count / elapsed  # days per second
                remaining_days = total_days - day_count
                eta_seconds = remaining_days / rate if rate > 0 else 0
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "calculating..."

            # Progress bar
            bar_length = 40
            filled = int(bar_length * day_count / total_days)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"\r  [{bar}] {progress_pct:.1f}% | {day_count}/{total_days} days | {success_count} successful | ETA: {eta_str}  ", end='', flush=True)
            last_update = now

        current += timedelta(days=1)

    # Final update
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  ✓ Downloaded {success_count}/{day_count} days in {int(elapsed)}s ({success_count/elapsed:.1f} days/sec)\n")

    if not all_minute_bars:
        print(f"  ❌ No data found for {symbol}")
        return

    # Combine all minute data
    minute_df = pd.concat(all_minute_bars).sort_index()
    minute_df = minute_df[~minute_df.index.duplicated(keep='last')]

    print(f"  📊 Total minute bars: {len(minute_df):,}")
    print(f"  📅 Range: {minute_df.index.min()} to {minute_df.index.max()}")

    # Process each timeframe
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        print(f"\n🔧 Processing {timeframe} ({i}/{len(TIMEFRAMES)})...")

        # Resample to timeframe
        if timeframe == '1H':
            resample_rule = '60T'
        else:
            resample_rule = timeframe

        print(f"  ⏳ Resampling...", end='', flush=True)
        df_tf = minute_df.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        print(f"\r  ✓ Resampled: {len(df_tf):,} bars")

        # Calculate features
        print(f"  ⏳ Calculating {timeframe} features...", end='', flush=True)
        df_features = build_feature_frame(df_tf)

        if df_features is None or df_features.empty:
            print(f"\r  ❌ Feature calculation failed")
            continue

        feature_count = len(df_features.columns)
        print(f"\r  ✓ Calculated {feature_count} features for {len(df_features):,} bars")

        # Save to parquet
        output_dir = FEATURE_STORE_DIR / symbol
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{symbol}_{timeframe}.parquet"

        print(f"  ⏳ Saving to parquet...", end='', flush=True)
        df_features.to_parquet(output_file, compression='snappy')

        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"\r  ✓ Saved: {output_file.name} ({file_size_mb:.1f} MB)")

    print(f"\n✅ {symbol} complete!\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download historical data and build features')
    parser.add_argument('--start-date', default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', choices=list(SYMBOLS.keys()) + ['all'], default='all', help='Symbol to process')

    args = parser.parse_args()

    # Determine which symbols to process
    if args.symbol == 'all':
        symbols_to_process = list(SYMBOLS.keys())
    else:
        symbols_to_process = [args.symbol]

    print("\n" + "="*80)
    print("HISTORICAL DATA DOWNLOAD & FEATURE GENERATION")
    print("="*80)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Symbols: {', '.join(symbols_to_process)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print("="*80)

    # Process each symbol
    for symbol in symbols_to_process:
        download_and_process_symbol(symbol, args.start_date, args.end_date)

    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)
    print("\nFeature files saved to:")
    for symbol in symbols_to_process:
        print(f"  feature_store/{symbol}/")
    print()


if __name__ == "__main__":
    main()
