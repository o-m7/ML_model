#!/usr/bin/env python3
"""
Fetch historical forex data from Polygon S3 flat files.
"""

import os
import sys
import boto3
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
import io

load_dotenv()

# S3 Configuration
S3_ACCESS_KEY = os.getenv('POLYGON_S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('POLYGON_S3_SECRET_KEY')
S3_ENDPOINT = os.getenv('POLYGON_S3_ENDPOINT')
S3_BUCKET = os.getenv('POLYGON_S3_BUCKET')

if not all([S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT, S3_BUCKET]):
    print("❌ Missing S3 credentials in .env!")
    sys.exit(1)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=S3_ENDPOINT
)


def list_available_files(prefix='us_forex'):
    """List available files in S3 bucket."""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=20
        )
        
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            print(f"✅ Found {len(files)} files with prefix '{prefix}'")
            for f in files[:10]:
                print(f"   - {f}")
            return files
        else:
            print(f"⚠️  No files found with prefix '{prefix}'")
            return []
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []


def fetch_forex_bars_from_s3(symbol: str, timeframe: str, days_back: int = 30):
    """
    Fetch and aggregate forex data from Polygon S3.
    
    Args:
        symbol: Symbol like 'EURUSD', 'XAUUSD'
        timeframe: '5T', '15T', '30T', '1H', '4H'
        days_back: Number of days to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n{'='*80}")
    print(f"Fetching {symbol} {timeframe} from S3 (last {days_back} days)")
    print(f"{'='*80}\n")
    
    # Map symbol to Polygon format
    if not symbol.startswith('C:'):
        symbol = f'C:{symbol}'
    
    timeframe_map = {
        '5T': 5,
        '15T': 15,
        '30T': 30,
        '1H': 60,
        '4H': 240
    }
    
    minutes = timeframe_map.get(timeframe, 60)
    
    # Try to find pre-aggregated minute bars first
    # Polygon flat files structure: us_forex_aggs/YYYY/MM/YYYY-MM-DD.csv.gz
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    all_bars = []
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day_str = current_date.strftime('%Y-%m-%d')
        
        # Try minute aggregates first
        s3_key = f'us_forex_aggs/{year}/{month}/{day_str}.csv.gz'
        
        try:
            print(f"  Fetching {day_str}...", end=' ')
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            
            # Read gzipped CSV
            df = pd.read_csv(io.BytesIO(response['Body'].read()), compression='gzip')
            
            # Filter for symbol
            if 'ticker' in df.columns:
                df = df[df['ticker'] == symbol]
            
            if not df.empty:
                # Convert timestamp
                if 'window_start' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['window_start'], utc=True)
                elif 't' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                
                # Standardize column names
                col_map = {
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
                    'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
                }
                df = df.rename(columns=col_map)
                
                # Keep only OHLCV
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].set_index('timestamp')
                    all_bars.append(df)
                    print(f"✅ {len(df)} bars")
                else:
                    print(f"⚠️  Missing columns: {df.columns.tolist()}")
            else:
                print("⚠️  No data for symbol")
                
        except s3_client.exceptions.NoSuchKey:
            print("⚠️  File not found")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        current_date += timedelta(days=1)
    
    if not all_bars:
        print("\n❌ No data fetched!")
        return pd.DataFrame()
    
    # Combine all days
    result = pd.concat(all_bars).sort_index()
    
    # Resample to target timeframe if needed
    if minutes > 1:
        print(f"\n  Resampling to {timeframe}...")
        result = result.resample(f'{minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    print(f"\n✅ Total: {len(result)} bars")
    print(f"   Range: {result.index.min()} to {result.index.max()}")
    
    return result


if __name__ == "__main__":
    # Test S3 access
    print("Testing Polygon S3 access...\n")
    
    # List available files
    list_available_files('us_forex_aggs')
    
    # Try to fetch EURUSD 1H data
    df = fetch_forex_bars_from_s3('EURUSD', '1H', days_back=7)
    
    if not df.empty:
        print("\n✅ S3 access working!")
        print("\nSample data:")
        print(df.tail())
    else:
        print("\n❌ S3 access failed or no data available")

