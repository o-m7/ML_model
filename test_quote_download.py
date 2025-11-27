"""
Quick test: Download sample quotes to verify S3 path works
"""

import boto3
from botocore.config import Config
import pandas as pd
import gzip

session = boto3.Session(
    aws_access_key_id='4937f95b-db8b-4d7e-8d54-756a82d4976e',
    aws_secret_access_key='o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1',
)

s3 = session.client(
    's3',
    endpoint_url='https://files.massive.com',
    config=Config(signature_version='s3v4'),
)

# Test download
dates_to_test = [
    '2025-11-25',
    '2025-11-24',
    '2025-11-20',
    '2025-11-01',
    '2020-01-02',  # First business day of 2020
]

for date_str in dates_to_test:
    date_obj = pd.to_datetime(date_str)
    year = date_obj.year
    month = f"{date_obj.month:02d}"
    
    object_key = f"global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz"
    
    print(f"\nTesting: {object_key}")
    
    try:
        response = s3.get_object(Bucket='flatfiles', Key=object_key)
        with gzip.GzipFile(fileobj=response['Body']) as gzipfile:
            df = pd.read_csv(gzipfile)
        
        print(f"  ✓ Found! {len(df)} quotes")
        print(f"  Columns: {list(df.columns)}")
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    except Exception as e:
        print(f"  ✗ Not found: {e}")
