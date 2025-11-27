#!/usr/bin/env python3
"""Find available quote data on Polygon S3"""

import boto3
from botocore.config import Config
from datetime import datetime, timedelta
import pandas as pd

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

print("Checking for available quote data...")
print("=" * 70)

# Check from recent backwards
test_dates = [
    ('2025-11-25', '2025 Nov'),
    ('2025-11-01', '2025 Nov start'),
    ('2025-10-01', '2025 Oct'),
    ('2025-01-01', '2025 Jan'),
    ('2024-12-31', '2024 Dec'),
    ('2024-01-01', '2024 Jan'),
    ('2023-01-01', '2023 Jan'),
    ('2022-01-01', '2022 Jan'),
    ('2021-01-01', '2021 Jan'),
    ('2020-01-01', '2020 Jan'),
]

for date_str, label in test_dates:
    dt = pd.to_datetime(date_str)
    year = dt.year
    month = f"{dt.month:02d}"
    day = f"{dt.day:02d}"
    
    key = f"flatfiles/global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz"
    
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
        size_mb = response['ContentLength'] / (1024*1024)
        print(f"✓ {label:20} {date_str:12} [{size_mb:8.2f} MB]")
    except Exception as e:
        print(f"✗ {label:20} {date_str:12} [NOT FOUND]")

print("\n" + "=" * 70)
print("Testing recent month for available days...")

# List all files in Nov 2025
try:
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix='flatfiles/global_forex/quotes_v1/2025/11/',
        MaxKeys=50
    )
    
    if 'Contents' in response:
        print(f"Found {len(response['Contents'])} files in Nov 2025:")
        for obj in response['Contents'][:10]:
            key = obj['Key']
            filename = key.split('/')[-1]
            print(f"  • {filename}")
        if len(response['Contents']) > 10:
            print(f"  ... and {len(response['Contents']) - 10} more")
    else:
        print("No files found in Nov 2025")
except Exception as e:
    print(f"Error listing files: {e}")
