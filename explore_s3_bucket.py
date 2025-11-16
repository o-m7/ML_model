#!/usr/bin/env python3
"""
Explore Massive.com S3 bucket structure to find available data
"""
import os
import boto3
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

s3 = setup_s3_client()

print("Exploring Massive.com S3 bucket structure...\n")
print("="*80)

# List top-level prefixes in flatfiles
print("\n1. Top-level prefixes in 'flatfiles' bucket:")
print("-" * 80)
try:
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix='flatfiles/',
        Delimiter='/'
    )

    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            print(f"  📁 {prefix['Prefix']}")
    else:
        print("  No common prefixes found")
except Exception as e:
    print(f"  Error: {e}")

# Check global_forex specifically
print("\n2. Contents of 'flatfiles/global_forex/':")
print("-" * 80)
try:
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix='flatfiles/global_forex/',
        Delimiter='/'
    )

    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            print(f"  📁 {prefix['Prefix']}")
    else:
        print("  No common prefixes found")
except Exception as e:
    print(f"  Error: {e}")

# Check minute_aggs_v1
print("\n3. Contents of 'flatfiles/global_forex/minute_aggs_v1/':")
print("-" * 80)
try:
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix='flatfiles/global_forex/minute_aggs_v1/',
        Delimiter='/'
    )

    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            print(f"  📁 {prefix['Prefix']}")
    else:
        print("  No common prefixes found")

    # Also show any files directly in this prefix
    if 'Contents' in response:
        print("\n  Direct files:")
        for obj in response['Contents'][:10]:
            print(f"    📄 {obj['Key']}")
except Exception as e:
    print(f"  Error: {e}")

# Check what years are available
print("\n4. Available years in 'flatfiles/global_forex/minute_aggs_v1/':")
print("-" * 80)
try:
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix='flatfiles/global_forex/minute_aggs_v1/',
        Delimiter='/'
    )

    if 'CommonPrefixes' in response:
        years = [prefix['Prefix'].split('/')[-2] for prefix in response['CommonPrefixes']]
        print(f"  Years: {sorted(years)}")

        # Check one year for months
        if years:
            test_year = sorted(years)[-1]  # Most recent year
            print(f"\n  Months in {test_year}:")
            response = s3.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=f'flatfiles/global_forex/minute_aggs_v1/{test_year}/',
                Delimiter='/'
            )
            if 'CommonPrefixes' in response:
                months = [prefix['Prefix'].split('/')[-2] for prefix in response['CommonPrefixes']]
                print(f"    {sorted(months)}")

                # Check one month for actual files
                if months:
                    test_month = sorted(months)[-1]
                    print(f"\n  Sample files in {test_year}/{test_month}:")
                    response = s3.list_objects_v2(
                        Bucket=BUCKET_NAME,
                        Prefix=f'flatfiles/global_forex/minute_aggs_v1/{test_year}/{test_month}/',
                        MaxKeys=5
                    )
                    if 'Contents' in response:
                        for obj in response['Contents'][:5]:
                            print(f"    📄 {obj['Key']}")
                            print(f"       Size: {obj['Size']/1024/1024:.1f} MB")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
