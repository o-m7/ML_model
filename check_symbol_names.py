#!/usr/bin/env python3
"""
Check what symbols are actually in the Massive.com data files.
"""

import gzip
import boto3
import pandas as pd
from botocore.config import Config

# Setup S3
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

# Download a recent file to inspect
print("Downloading sample file to inspect...")
object_key = 'flatfiles/global_forex/minute_aggs_v1/2024/11/2024-11-01.csv.gz'
if object_key.startswith(bucket + '/'):
    object_key = object_key[len(bucket + '/'):]

local_file = 'sample_data.csv.gz'

try:
    s3.download_file(bucket, object_key, local_file)
    print(f"✓ Downloaded {object_key}\n")

    # Read and inspect
    with gzip.open(local_file, 'rt') as f:
        df = pd.read_csv(f, nrows=1000)  # Just first 1000 rows

    print("=== FILE STRUCTURE ===")
    print(f"Columns: {list(df.columns)}\n")

    # Check what symbol column is called
    symbol_col = None
    for col in ['ticker', 'symbol', 'pair', 'instrument']:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col:
        print(f"Symbol column: '{symbol_col}'\n")

        # Show unique symbols (metal-related)
        symbols = df[symbol_col].unique()

        # Filter for gold/silver
        metal_symbols = [s for s in symbols if any(x in str(s).upper() for x in ['XAU', 'XAG', 'GOLD', 'SILVER'])]

        if metal_symbols:
            print("=== METAL SYMBOLS FOUND ===")
            for sym in sorted(metal_symbols):
                count = len(df[df[symbol_col] == sym])
                print(f"  {sym}: {count} rows")
        else:
            print("=== ALL SYMBOLS (first 20) ===")
            for sym in sorted(symbols)[:20]:
                count = len(df[df[symbol_col] == sym])
                print(f"  {sym}: {count} rows")
            print(f"  ... ({len(symbols)} total symbols)")
    else:
        print("⚠️ No symbol column found!")
        print("\nFirst few rows:")
        print(df.head())

    # Show sample row
    print("\n=== SAMPLE ROW ===")
    print(df.iloc[0])

    import os
    os.unlink(local_file)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
