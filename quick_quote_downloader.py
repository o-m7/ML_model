#!/usr/bin/env python3
"""
quick_quote_downloader.py - Fast quote downloader for date range
"""

import boto3
from botocore.config import Config
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import gzip
import sys

# S3 credentials
session = boto3.Session(
    aws_access_key_id='4937f95b-db8b-4d7e-8d54-756a82d4976e',
    aws_secret_access_key='o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1',
)

s3 = session.client(
    's3',
    endpoint_url='https://files.massive.com',
    config=Config(signature_version='s3v4'),
)

bucket = 'flatfiles'
start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2025-11-25')

print(f"Downloading quotes from {start_date.date()} to {end_date.date()}")
print("=" * 70)

all_quotes = []
current = start_date
day_count = 0
success_count = 0
fail_count = 0

while current <= end_date:
    date_str = current.strftime("%Y-%m-%d")
    year = current.year
    month = f"{current.month:02d}"
    day = f"{current.day:02d}"
    
    key = f"flatfiles/global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz"
    
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        with gzip.GzipFile(fileobj=response['Body']) as f:
            df = pd.read_csv(f)
        
        # Parse timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        if 'quote_at' in df.columns:
            df['quote_at'] = pd.to_datetime(df['quote_at'], unit='ns')
        
        all_quotes.append(df)
        success_count += 1
        
        if success_count % 50 == 0:
            print(f"[{success_count}] Downloaded {date_str}: {len(df)} quotes")
    
    except Exception as e:
        fail_count += 1
    
    day_count += 1
    if day_count % 100 == 0:
        pct = (day_count / ((end_date - start_date).days + 1)) * 100
        print(f"Progress: {day_count} days ({pct:.1f}%), {success_count} successful")
    
    current += timedelta(days=1)

print("=" * 70)
print(f"Download complete: {success_count} successful, {fail_count} failed")

if all_quotes:
    df_all = pd.concat(all_quotes, ignore_index=True)
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Total quotes: {len(df_all):,}")
    print(f"Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
    print(f"Columns: {list(df_all.columns)}")
    
    # Save intermediate
    output_dir = Path("feature_store/C:XAU-USD/quotes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw parquet
    raw_file = output_dir / "quotes_raw.parquet"
    df_all.to_parquet(raw_file, compression='snappy', index=False)
    print(f"\nSaved: {raw_file}")
    
    # Aggregate to timeframes
    for tf in ['1T', '5T', '15T', '30T']:
        print(f"\nAggregating to {tf}...")
        df_idx = df_all.set_index('timestamp')
        
        ohlc = df_idx['mid_price'].resample(tf).ohlc()
        volume = df_idx.resample(tf).size()
        
        agg = df_idx.resample(tf).agg({
            'bid': ['first', 'last', 'min', 'max'],
            'ask': ['first', 'last', 'min', 'max'],
            'mid_price': ['mean', 'std'],
            'bid_size': ['sum', 'mean', 'max'],
            'ask_size': ['sum', 'mean', 'max'],
        })
        
        agg.columns = ['_'.join(col).strip() for col in agg.columns]
        
        result = ohlc.copy()
        result['volume'] = volume
        for col in agg.columns:
            result[col] = agg[col]
        
        result = result.reset_index()
        
        tf_file = output_dir / f"C:XAU-USD_{tf}_quotes.parquet"
        result.to_parquet(tf_file, compression='snappy', index=False)
        print(f"  ✓ {tf}: {len(result)} bars → {tf_file}")
    
    print("\n✅ All files saved!")
else:
    print("No quotes downloaded!")
    sys.exit(1)
