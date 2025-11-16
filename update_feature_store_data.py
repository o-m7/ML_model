#!/usr/bin/env python3
"""
Update feature store with new data.
If you have downloaded data, this will help you save it to the feature store.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def update_feature_store(df, symbol="XAUUSD", timeframe="15T"):
    """Save data to feature store."""

    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if df.columns[0] in ['index', 'date', 'datetime']:
                df = df.rename(columns={df.columns[0]: 'timestamp'})
        else:
            print("❌ No timestamp found!")
            return False

    # Convert to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Remove duplicates
    df = df.drop_duplicates(subset='timestamp', keep='last')

    # Ensure required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False

    # Keep only required columns
    df = df[required_cols]

    # Create feature store directory
    feature_store = Path("feature_store") / symbol
    feature_store.mkdir(parents=True, exist_ok=True)

    # Save
    output_file = feature_store / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(output_file, index=False)

    # Report
    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"✅ Saved to {output_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Days: {date_range} ({date_range/30:.1f} months, {date_range/365:.1f} years)")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

    return True


# Example usage - modify these paths to your downloaded data
if __name__ == "__main__":
    print("=" * 80)
    print("UPDATE FEATURE STORE WITH NEW DATA")
    print("=" * 80)

    # OPTION 1: If you have a CSV file
    # df = pd.read_csv("path/to/your/downloaded_data.csv")

    # OPTION 2: If you have a parquet file
    # df = pd.read_parquet("path/to/your/downloaded_data.parquet")

    # OPTION 3: Check what files you have
    print("\n📂 Looking for data files in current directory...")
    import os

    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'XAUUSD' in f.upper()]
    parquet_files = [f for f in os.listdir('.') if f.endswith('.parquet') and 'XAUUSD' in f.upper()]

    if csv_files:
        print(f"\n   Found CSV files: {csv_files}")
        print(f"\n   To load, uncomment and modify:")
        print(f"   df = pd.read_csv('{csv_files[0]}')")

    if parquet_files:
        print(f"\n   Found Parquet files: {parquet_files}")
        print(f"\n   To load, uncomment and modify:")
        print(f"   df = pd.read_parquet('{parquet_files[0]}')")

    if not csv_files and not parquet_files:
        print("\n   ❌ No XAUUSD data files found in current directory")
        print("\n   Please specify the path to your downloaded data:")
        print("   1. Edit this file (update_feature_store_data.py)")
        print("   2. Uncomment the appropriate df = pd.read_csv/parquet line")
        print("   3. Replace with your actual file path")
        print("   4. Run: python update_feature_store_data.py")
