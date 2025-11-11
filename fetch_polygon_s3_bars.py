#!/usr/bin/env python3
"""
Fetch recent forex bars from Polygon S3 for live signal generation.
Replaces the broken API fetching with direct S3 access.
"""

import os
import boto3
import pandas as pd
import gzip
from datetime import datetime, timedelta, timezone
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# S3 Configuration
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('POLYGON_S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('POLYGON_S3_SECRET_KEY'),
    endpoint_url=os.getenv('POLYGON_S3_ENDPOINT')
)

S3_BUCKET = os.getenv('POLYGON_S3_BUCKET', 'flatfiles')

# Ticker mapping
TICKER_MAP = {
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD',
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
}

TIMEFRAME_MINUTES = {'5T': 5, '15T': 15, '30T': 30, '1H': 60, '4H': 240}


def fetch_s3_bars(symbol: str, timeframe: str, bars_needed: int = 200) -> pd.DataFrame:
    """
    Fetch recent bars from Polygon S3 minute aggregates.
    
    Args:
        symbol: e.g., 'EURUSD', 'XAUUSD'
        timeframe: '5T', '15T', '30T', '1H', '4H'
        bars_needed: Number of bars to return
    
    Returns:
        DataFrame with OHLCV data, indexed by timestamp
    """
    polygon_ticker = TICKER_MAP.get(symbol, f'C:{symbol}')
    minutes = TIMEFRAME_MINUTES[timeframe]
    
    # Calculate how many days back we need
    # For 4H bars, 200 bars = 800 hours = ~33 days (with weekends = ~47 days)
    days_back = int((bars_needed * minutes) / (60 * 24) * 1.5) + 7  # Add buffer for weekends
    
    all_bars = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    current_date = start_date
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Construct S3 key for minute aggregates
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        date_str = current_date.strftime('%Y-%m-%d')
        
        file_key = f'global_forex/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'
        
        try:
            # Fetch file from S3
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
            compressed_data = response['Body'].read()
            
            # Decompress and read CSV
            with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
                df_day = pd.read_csv(gz)
            
            # Filter for specific ticker
            if 'ticker' in df_day.columns:
                df_day = df_day[df_day['ticker'] == polygon_ticker].copy()
            
            if not df_day.empty:
                # Convert timestamp (Polygon uses milliseconds)
                if 'window_start' in df_day.columns:
                    df_day['timestamp'] = pd.to_datetime(df_day['window_start'], unit='ms', utc=True)
                elif 'timestamp' in df_day.columns:
                    df_day['timestamp'] = pd.to_datetime(df_day['timestamp'], unit='ms', utc=True)
                
                # Rename columns to standard OHLCV
                df_day = df_day.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'transactions': 'volume'  # Fallback if volume not present
                })
                
                df_day = df_day[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                df_day = df_day.set_index('timestamp').sort_index()
                
                # Resample to target timeframe if needed
                if minutes > 1:
                    df_day = df_day.resample(timeframe).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                
                all_bars.append(df_day)
                
        except s3_client.exceptions.NoSuchKey:
            pass  # File doesn't exist for this date
        except Exception as e:
            print(f"  ⚠️  Error fetching {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    if not all_bars:
        return None
    
    # Combine all days
    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep='last')]  # Remove duplicates
    
    # Return only the requested number of bars
    return result.tail(bars_needed)


if __name__ == "__main__":
    # Test
    print("Testing Polygon S3 bar fetching...")
    
    df = fetch_s3_bars('EURUSD', '1H', bars_needed=100)
    
    if df is not None and not df.empty:
        print(f"\n✅ SUCCESS! Fetched {len(df)} bars")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nSample data:")
        print(df.tail())
    else:
        print("\n❌ Failed to fetch data")

