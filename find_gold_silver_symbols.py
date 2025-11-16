#!/usr/bin/env python3
"""
Find all gold/silver related symbols in Massive.com data
"""
import gzip
import os
from datetime import datetime

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# AWS credentials
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', '4937f95b-db8b-4d7e-8d54-756a82d4976e')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1')
ENDPOINT_URL = 'https://files.massive.com'
BUCKET_NAME = 'flatfiles'

def setup_s3_client():
    """Initialize S3 client."""
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    return session.client('s3', endpoint_url=ENDPOINT_URL)

# Test multiple recent dates
test_dates = ['2024-11-01', '2024-10-01', '2024-09-01', '2023-01-15', '2022-06-15']

s3 = setup_s3_client()

print("Searching for gold/silver symbols in Massive.com data...\n")

for date_str in test_dates:
    print(f"\nChecking {date_str}...")

    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')

    object_key = f'flatfiles/global_forex/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'
    temp_file = f"/tmp/{date_str}.csv.gz"

    try:
        # Download file
        s3.download_file(BUCKET_NAME, object_key, temp_file)

        # Read and search
        with gzip.open(temp_file, 'rt') as f:
            df = pd.read_csv(f)

        # Find gold/silver symbols
        if 'ticker' in df.columns:
            all_symbols = df['ticker'].unique()

            # Search for all XAU/XAG patterns
            gold_symbols = [s for s in all_symbols if 'XAU' in str(s)]
            silver_symbols = [s for s in all_symbols if 'XAG' in str(s)]

            print(f"  Total symbols: {len(all_symbols)}")

            # Check specific formats
            formats_to_check = [
                'XAUUSD', 'XAU-USD', 'XAU/USD', 'C:XAUUSD', 'C:XAU-USD', 'C:XAU/USD',
                'XAGUSD', 'XAG-USD', 'XAG/USD', 'C:XAGUSD', 'C:XAG-USD', 'C:XAG/USD'
            ]

            found_formats = [fmt for fmt in formats_to_check if fmt in all_symbols]

            if found_formats:
                print(f"  ✓ EXACT MATCHES: {found_formats}")

            if gold_symbols:
                print(f"  ✓ All gold symbols (XAU): {gold_symbols}")
            else:
                print(f"  ✗ No gold symbols (XAU)")

            if silver_symbols:
                print(f"  ✓ All silver symbols (XAG): {silver_symbols}")
            else:
                print(f"  ✗ No silver symbols (XAG)")

            # Show a few random symbols as reference
            print(f"  Sample symbols: {list(all_symbols[:10])}")

        # Cleanup
        os.unlink(temp_file)

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nIf no XAU/XAG symbols found, possible reasons:")
print("1. Massive.com may not have gold/silver spot FX data")
print("2. Data might be in a different bucket/path")
print("3. Symbols might use different naming (e.g., GOLD, XAUUSD without separators)")
print("\nCheck Massive.com documentation for available symbols.")
