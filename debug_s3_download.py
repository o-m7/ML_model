#!/usr/bin/env python3
"""
Debug script to see exactly what's in the S3 files
"""
import os
import boto3
import pandas as pd
import gzip
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# S3 Configuration
ACCESS_KEY_ID = os.getenv('Access_Key_ID')
SECRET_ACCESS_KEY = os.getenv('Secret_Access_Key')
S3_ENDPOINT = os.getenv('S3_endpoint')
BUCKET = os.getenv('Bucket')

print(f"S3 Endpoint: {S3_ENDPOINT}")
print(f"Bucket: {BUCKET}")
print(f"Access Key: {ACCESS_KEY_ID[:10]}...")
print()

# Initialize S3
session = boto3.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
)
s3 = session.client('s3', endpoint_url=S3_ENDPOINT)

# Test a specific date that should have data
test_dates = ['2024-11-14', '2024-11-13', '2024-11-12', '2023-01-15', '2022-06-15']

for test_date in test_dates:
    print(f"\n{'='*80}")
    print(f"Testing date: {test_date}")
    print('='*80)

    year = test_date[:4]
    month = test_date[5:7]

    file_key = f'global_forex/minute_aggs_v1/{year}/{month}/{test_date}.csv.gz'
    print(f"S3 Key: {file_key}")

    try:
        # Download
        print("Downloading...", end=' ', flush=True)
        response = s3.get_object(Bucket=BUCKET, Key=file_key)
        compressed_data = response['Body'].read()
        print(f"✓ Got {len(compressed_data)} bytes")

        # Decompress
        print("Decompressing...", end=' ', flush=True)
        with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
            df = pd.read_csv(gz)
        print(f"✓ Got {len(df)} rows")

        # Show structure
        print(f"\nColumns: {list(df.columns)}")
        print(f"Shape: {df.shape}")

        # Check for ticker/symbol column
        if 'ticker' in df.columns:
            print(f"\n✓ Found 'ticker' column")
            unique_tickers = df['ticker'].unique()
            print(f"Total unique tickers: {len(unique_tickers)}")

            # Search for XAU/XAG
            gold = [t for t in unique_tickers if 'XAU' in str(t)]
            silver = [t for t in unique_tickers if 'XAG' in str(t)]

            if gold:
                print(f"✓ GOLD SYMBOLS: {gold}")
            else:
                print(f"✗ No gold symbols found")

            if silver:
                print(f"✓ SILVER SYMBOLS: {silver}")
            else:
                print(f"✗ No silver symbols found")

            print(f"\nSample tickers: {list(unique_tickers[:20])}")

            # Try to filter for gold
            if gold:
                gold_symbol = gold[0]
                print(f"\n\nTrying to filter for: {gold_symbol}")
                df_gold = df[df['ticker'] == gold_symbol].copy()
                print(f"Found {len(df_gold)} rows for {gold_symbol}")

                if len(df_gold) > 0:
                    print(f"\nSample data:")
                    print(df_gold.head(3))
                    print(f"\nColumn data types:")
                    print(df_gold.dtypes)

        # Success - we found working data!
        print("\n✅ SUCCESS - Found working data file!")
        break

    except s3.exceptions.NoSuchKey:
        print(f"✗ File not found")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*80)
