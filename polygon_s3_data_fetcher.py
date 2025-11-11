#!/usr/bin/env python3
"""
Fetch historical forex data from Polygon S3 flat files.
Uses boto3 to access S3-compatible storage at files.massive.com.
"""

import os
import sys
import boto3
import pandas as pd
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

# S3 Configuration
S3_ACCESS_KEY = os.getenv('POLYGON_S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('POLYGON_S3_SECRET_KEY')
S3_ENDPOINT = os.getenv('POLYGON_S3_ENDPOINT')
S3_BUCKET = os.getenv('POLYGON_S3_BUCKET')

if not all([S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT, S3_BUCKET]):
    print("❌ Missing S3 credentials in .env file!")
    sys.exit(1)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=S3_ENDPOINT
)

# Polygon ticker mapping
TICKER_MAP = {
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD',
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
}

OUTPUT_DIR = Path(__file__).parent / 'polygon_data'
OUTPUT_DIR.mkdir(exist_ok=True)


def list_available_files(prefix='us_forex'):
    """List available files in the S3 bucket."""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=100
        )
        
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            print(f"✅ Found {len(files)} files with prefix '{prefix}'")
            return files
        else:
            print(f"⚠️  No files found with prefix '{prefix}'")
            return []
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []


def fetch_file_from_s3(file_key: str) -> pd.DataFrame:
    """
    Fetch and parse a gzipped CSV file from S3.
    
    Args:
        file_key: S3 object key (e.g., 'us_forex_quotes/2025/11/2025-11-08.csv.gz')
    
    Returns:
        DataFrame with the file contents
    """
    try:
        print(f"  Fetching: {file_key}")
        
        # Download from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        
        # Read and decompress
        compressed_data = response['Body'].read()
        
        with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
            df = pd.read_csv(gz)
        
        print(f"  ✅ Got {len(df):,} rows")
        return df
        
    except s3_client.exceptions.NoSuchKey:
        print(f"  ⚠️  File not found: {file_key}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return pd.DataFrame()


def aggregate_to_bars(df: pd.DataFrame, timeframe: str, symbol: str) -> pd.DataFrame:
    """
    Convert tick/quote data to OHLCV bars.
    
    Args:
        df: Raw tick data
        timeframe: '5T', '15T', '30T', '1H', '4H'
        symbol: e.g., 'C:EURUSD'
    
    Returns:
        OHLCV DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter for specific symbol if present
    if 'ticker' in df.columns:
        df = df[df['ticker'] == symbol].copy()
    
    if df.empty:
        print(f"  ⚠️  No data for {symbol} in this file")
        return pd.DataFrame()
    
    # Convert timestamp to datetime (Polygon uses nanoseconds)
    if 'sip_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns', utc=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
    else:
        print(f"  ❌ No timestamp column found. Columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    df = df.set_index('timestamp').sort_index()
    
    # Determine price column (bid/ask midpoint is most accurate for forex)
    if 'bid_price' in df.columns and 'ask_price' in df.columns:
        df['price'] = (df['bid_price'] + df['ask_price']) / 2
    elif 'price' in df.columns:
        df['price'] = df['price']
    else:
        print(f"  ❌ No price columns found. Columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    # Aggregate to timeframe
    ohlcv = df.resample(timeframe).agg({
        'price': ['first', 'max', 'min', 'last', 'count']
    })
    
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv = ohlcv.dropna()
    
    return ohlcv


def fetch_historical_data(symbol: str, timeframe: str, days_back: int = 30):
    """
    Fetch historical data for a symbol and aggregate to specified timeframe.
    
    Args:
        symbol: e.g., 'EURUSD', 'XAUUSD'
        timeframe: '5T', '15T', '30T', '1H', '4H'
        days_back: Number of days of historical data to fetch
    
    Returns:
        OHLCV DataFrame
    """
    polygon_ticker = TICKER_MAP.get(symbol, f'C:{symbol}')
    
    print(f"\n{'='*80}")
    print(f"Fetching {symbol} ({polygon_ticker}) - {timeframe} - Last {days_back} days")
    print(f"{'='*80}\n")
    
    all_bars = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    current_date = start_date
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Construct S3 key (check both quotes and aggregates)
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Try quotes first (tick data)
        file_key = f'us_forex_quotes/{year}/{month}/{date_str}.csv.gz'
        df_day = fetch_file_from_s3(file_key)
        
        if not df_day.empty:
            bars = aggregate_to_bars(df_day, timeframe, polygon_ticker)
            if not bars.empty:
                all_bars.append(bars)
        
        current_date += timedelta(days=1)
    
    if not all_bars:
        print("\n❌ No data fetched!")
        return pd.DataFrame()
    
    # Combine all days
    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep='last')]  # Remove duplicates
    
    # Save to parquet
    output_file = OUTPUT_DIR / f"{symbol}_{timeframe}.parquet"
    result.to_parquet(output_file)
    
    print(f"\n✅ SUCCESS!")
    print(f"   Bars fetched: {len(result):,}")
    print(f"   Date range: {result.index.min()} to {result.index.max()}")
    print(f"   Saved to: {output_file}")
    
    return result


def main():
    """Test the S3 fetcher."""
    print("="*80)
    print("POLYGON S3 FLAT FILE DATA FETCHER")
    print("="*80)
    
    # First, list available files to verify access
    print("\nChecking S3 access...")
    files = list_available_files('us_forex_quotes/2025/11')
    
    if files:
        print(f"\nSample files:")
        for f in files[:5]:
            print(f"  - {f}")
    
    # Fetch data for EURUSD 1H (last 7 days)
    print("\n" + "="*80)
    print("FETCHING TEST DATA: EURUSD 1H")
    print("="*80)
    
    df = fetch_historical_data('EURUSD', '1H', days_back=7)
    
    if not df.empty:
        print("\n" + "="*80)
        print("SAMPLE DATA:")
        print("="*80)
        print(df.tail(10))


if __name__ == "__main__":
    main()
