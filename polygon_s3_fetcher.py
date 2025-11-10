#!/usr/bin/env python3
"""
Fetch historical forex data from Polygon S3 flat files.
Polygon provides historical forex data via S3 at:
https://polygon.io/flat-files
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import gzip
import io

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
OUTPUT_DIR = Path(__file__).parent / 'polygon_data'
OUTPUT_DIR.mkdir(exist_ok=True)

# Polygon S3 flat file format
# Official docs: https://polygon.io/docs/flat-files/forex/overview
# File Browser: https://polygon.io/flat-files
FLAT_FILE_BASE = "https://files.polygon.io"

def fetch_forex_flat_file(date: datetime, pair: str = "C:EURUSD", data_type: str = "quotes"):
    """
    Fetch forex data from Polygon flat files for a specific date.
    
    Polygon flat files structure:
    https://files.polygon.io/us_forex_{data_type}/{YYYY}/{MM}/YYYY-MM-DD.csv.gz
    
    data_type: 'quotes' (tick data) or 'agg' (pre-aggregated bars)
    
    Note: Requires Polygon API key with flat file access.
    """
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day_str = date.strftime('%Y-%m-%d')
    
    # Try aggregates first (pre-built 1-minute bars), fall back to quotes
    url = f"{FLAT_FILE_BASE}/us_forex_{data_type}/{year}/{month}/{day_str}.csv.gz"
    
    print(f"  Fetching {day_str}...")
    print(f"  URL: {url}")
    
    try:
        headers = {'Authorization': f'Bearer {POLYGON_API_KEY}'}
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Decompress gzip
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                df = pd.read_csv(gz)
            
            # Filter for specific pair if provided
            if pair and 'ticker' in df.columns:
                df = df[df['ticker'] == pair]
            
            print(f"  ✅ Got {len(df)} rows for {day_str}")
            return df
        elif response.status_code == 404:
            print(f"  ⚠️  No data available for {day_str}")
            return None
        else:
            print(f"  ❌ Error {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error fetching {day_str}: {e}")
        return None


def aggregate_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggregate tick/quote data into OHLCV bars.
    
    Args:
        df: DataFrame with tick data (timestamp, price, volume)
        timeframe: '5T', '15T', '30T', '1H', '4H'
    
    Returns:
        OHLCV DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Ensure timestamp column
    if 'timestamp' not in df.columns and 'sip_timestamp' in df.columns:
        df['timestamp'] = df['sip_timestamp']
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
    df = df.set_index('timestamp').sort_index()
    
    # Use price columns (adjust based on actual Polygon flat file schema)
    price_col = 'price' if 'price' in df.columns else 'bid_price' if 'bid_price' in df.columns else 'ask_price'
    
    if price_col not in df.columns:
        print(f"  ⚠️  No price column found. Available: {df.columns.tolist()}")
        return pd.DataFrame()
    
    # Resample to timeframe
    ohlcv = df.resample(timeframe).agg({
        price_col: ['first', 'max', 'min', 'last'],
        'size': 'sum' if 'size' in df.columns else 'count'
    })
    
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv = ohlcv.dropna()
    
    return ohlcv


def fetch_historical_range(start_date: datetime, end_date: datetime, pair: str, timeframe: str):
    """
    Fetch and aggregate historical forex data for a date range.
    
    Args:
        start_date: Start date
        end_date: End date
        pair: Forex pair (e.g., 'C:EURUSD')
        timeframe: Target timeframe ('5T', '15T', '30T', '1H', '4H')
    
    Returns:
        OHLCV DataFrame
    """
    print(f"\n{'='*80}")
    print(f"Fetching {pair} {timeframe} from {start_date.date()} to {end_date.date()}")
    print(f"{'='*80}\n")
    
    all_dfs = []
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends (forex is closed Sat/Sun)
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            df_day = fetch_forex_flat_file(current_date, pair)
            
            if df_day is not None and not df_day.empty:
                # Aggregate to timeframe
                ohlcv = aggregate_to_timeframe(df_day, timeframe)
                if not ohlcv.empty:
                    all_dfs.append(ohlcv)
        
        current_date += timedelta(days=1)
    
    if not all_dfs:
        print("\n❌ No data fetched!")
        return pd.DataFrame()
    
    # Combine all days
    result = pd.concat(all_dfs).sort_index()
    
    # Save to parquet
    symbol = pair.replace('C:', '')
    output_file = OUTPUT_DIR / f"{symbol}_{timeframe}.parquet"
    result.to_parquet(output_file)
    
    print(f"\n✅ Fetched {len(result)} bars")
    print(f"   Saved to: {output_file}")
    print(f"   Date range: {result.index.min()} to {result.index.max()}")
    
    return result


def main():
    """Test fetcher with recent data."""
    # Fetch last 7 days of EURUSD 1H data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    df = fetch_historical_range(
        start_date=start_date,
        end_date=end_date,
        pair='C:EURUSD',
        timeframe='1H'
    )
    
    if not df.empty:
        print("\nSample data:")
        print(df.tail())


if __name__ == "__main__":
    main()

