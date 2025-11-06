#!/usr/bin/env python3
"""
UPDATE FEATURE STORE WITH LATEST DATA
======================================

Fetches recent data from Polygon and appends to existing parquet files.
Runs feature engineering and updates all timeframes.

Usage:
    python3 update_feature_store.py --symbol XAUUSD
    python3 update_feature_store.py --symbol XAUUSD --start-date 2024-10-23
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Add Polygon pipeline to path
sys.path.insert(0, str(Path.home() / "Desktop" / "Polygon-ML-data"))

try:
    from polygon_forex_pipeline import PolygonForexPipeline
    from feature_engineering import ForexFeatureEngineer
    POLYGON_AVAILABLE = True
except ImportError:
    print("⚠️  Polygon pipeline not found. Will use manual update method.")
    POLYGON_AVAILABLE = False


class FeatureStoreUpdater:
    """Updates feature store with latest Polygon data."""
    
    def __init__(self, feature_store_path: Path):
        self.feature_store_path = Path(feature_store_path)
        self.timeframes = ['1T', '15T', '30T', '1H', '4H']
        
    def get_last_date(self, symbol: str, timeframe: str) -> datetime:
        """Get the last date in existing parquet file."""
        file_path = self.feature_store_path / symbol / f"{symbol}_{timeframe}.parquet"
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
        
        print(f"\n📂 Checking {file_path.name}...")
        df = pd.read_parquet(file_path)
        
        # Find timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            last_date = df['timestamp'].max()
        elif isinstance(df.index, pd.DatetimeIndex):
            last_date = df.index.max()
        else:
            print(f"❌ Cannot find timestamp in {file_path}")
            return None
        
        print(f"  Last date: {last_date}")
        print(f"  Total rows: {len(df):,}")
        
        return last_date
    
    def fetch_new_data(self, symbol: str, start_date: datetime, end_date: datetime = None):
        """Fetch new data from Polygon."""
        if end_date is None:
            end_date = datetime.now()
        
        print(f"\n{'='*80}")
        print(f"FETCHING NEW DATA FROM POLYGON")
        print(f"{'='*80}")
        print(f"Symbol: {symbol}")
        print(f"Start: {start_date.strftime('%Y-%m-%d')}")
        print(f"End: {end_date.strftime('%Y-%m-%d')}")
        print(f"Days to fetch: {(end_date - start_date).days}")
        
        if not POLYGON_AVAILABLE:
            print("\n❌ Polygon pipeline not available.")
            print("   Please run from /Users/omar/Desktop/Polygon-ML-data")
            return None
        
        try:
            # Initialize pipeline
            pipeline = PolygonForexPipeline()
            
            # Fetch data for the date range
            print(f"\nFetching minute data...")
            # This will depend on your pipeline's exact methods
            # You may need to adjust this call
            
            # For now, show manual instructions
            print("\n📝 MANUAL STEPS REQUIRED:")
            print(f"   1. cd /Users/omar/Desktop/Polygon-ML-data")
            print(f"   2. python3 polygon_forex_pipeline.py --mode backfill \\")
            print(f"      --start-date {start_date.strftime('%Y-%m-%d')} \\")
            print(f"      --end-date {end_date.strftime('%Y-%m-%d')} \\")
            print(f"      --symbol {symbol}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def append_to_parquet(self, symbol: str, timeframe: str, new_data: pd.DataFrame):
        """Append new data to existing parquet file."""
        file_path = self.feature_store_path / symbol / f"{symbol}_{timeframe}.parquet"
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        print(f"\n📊 Updating {file_path.name}...")
        
        # Load existing data
        existing_df = pd.read_parquet(file_path)
        print(f"  Existing rows: {len(existing_df):,}")
        
        # Ensure timestamp column
        if 'timestamp' not in new_data.columns and isinstance(new_data.index, pd.DatetimeIndex):
            new_data = new_data.reset_index()
            new_data.rename(columns={'index': 'timestamp'}, inplace=True)
        
        if 'timestamp' not in existing_df.columns and isinstance(existing_df.index, pd.DatetimeIndex):
            existing_df = existing_df.reset_index()
            existing_df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # Combine
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        
        # Remove duplicates (keep last)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp')
        combined_df = combined_df.drop_duplicates(subset='timestamp', keep='last')
        
        print(f"  New rows added: {len(new_data):,}")
        print(f"  Total rows: {len(combined_df):,}")
        print(f"  New date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        
        # Save
        backup_path = file_path.with_suffix('.parquet.backup')
        print(f"  Creating backup: {backup_path.name}")
        file_path.rename(backup_path)
        
        try:
            combined_df.to_parquet(file_path, index=False)
            print(f"  ✅ Updated successfully!")
            
            # Remove backup if successful
            backup_path.unlink()
            
        except Exception as e:
            print(f"  ❌ Error saving: {e}")
            # Restore backup
            backup_path.rename(file_path)
            return False
        
        return True
    
    def update_all_timeframes(self, symbol: str, start_date: datetime = None):
        """Update all timeframes for a symbol."""
        print(f"\n{'='*80}")
        print(f"UPDATING ALL TIMEFRAMES FOR {symbol}")
        print(f"{'='*80}")
        
        # Get last date from base timeframe (1T)
        if start_date is None:
            last_date = self.get_last_date(symbol, '1T')
            if last_date is None:
                print("❌ Cannot determine last date")
                return False
            
            # Start from day after last date
            start_date = last_date + timedelta(days=1)
        
        print(f"\n📅 Update will start from: {start_date.strftime('%Y-%m-%d')}")
        
        # Make datetime.now() timezone-aware for comparison
        now = datetime.now(timezone.utc) if start_date.tzinfo else datetime.now()
        print(f"📅 Update will go to: {now.strftime('%Y-%m-%d')}")
        
        days_to_add = (now - start_date).days
        
        if days_to_add <= 0:
            print(f"\n✅ Data is already up to date!")
            return True
        
        print(f"\n⏳ Need to add {days_to_add} days of data")
        
        # Show instructions
        print(f"\n{'='*80}")
        print("TO UPDATE YOUR FEATURE STORE:")
        print(f"{'='*80}")
        print(f"\n1️⃣  Fetch new raw data from Polygon:")
        print(f"   cd /Users/omar/Desktop/Polygon-ML-data")
        print(f"   python3 polygon_forex_pipeline.py --mode backfill \\")
        print(f"       --start-date {start_date.strftime('%Y-%m-%d')} \\")
        print(f"       --end-date {datetime.now().strftime('%Y-%m-%d')} \\")
        print(f"       --symbol {symbol}")
        
        print(f"\n2️⃣  Process and add to feature store:")
        print(f"   cd /Users/omar/Desktop/Polygon-ML-data")
        print(f"   python3 feature_engineering.py \\")
        print(f"       --input raw_data/{symbol}_minute.csv \\")
        print(f"       --output /Users/omar/Desktop/ML_Trading/feature_store/{symbol}")
        
        print(f"\n3️⃣  Verify update:")
        print(f"   python3 update_feature_store.py --symbol {symbol} --verify")
        
        return True


def verify_update(symbol: str, feature_store_path: Path):
    """Verify that data is up to date."""
    print(f"\n{'='*80}")
    print("VERIFYING FEATURE STORE")
    print(f"{'='*80}")
    
    updater = FeatureStoreUpdater(feature_store_path)
    timeframes = ['1T', '15T', '30T', '1H', '4H']
    
    all_good = True
    
    for tf in timeframes:
        last_date = updater.get_last_date(symbol, tf)
        if last_date is None:
            continue
        
        days_behind = (datetime.now() - last_date.replace(tzinfo=None)).days
        
        if days_behind == 0:
            print(f"  ✅ {tf}: Up to date!")
        elif days_behind <= 1:
            print(f"  ✅ {tf}: 1 day behind (acceptable)")
        else:
            print(f"  ⚠️  {tf}: {days_behind} days behind")
            all_good = False
    
    print()
    if all_good:
        print("✅ ✅ ✅  ALL TIMEFRAMES ARE UP TO DATE!")
    else:
        print("⚠️  SOME TIMEFRAMES NEED UPDATING")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(description='Update feature store with latest data')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol to update')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--verify', action='store_true', help='Verify data is up to date')
    parser.add_argument('--feature-store', type=str, 
                       default='/Users/omar/Desktop/ML_Trading/feature_store',
                       help='Path to feature store')
    
    args = parser.parse_args()
    
    feature_store_path = Path(args.feature_store)
    
    if args.verify:
        verify_update(args.symbol, feature_store_path)
        return 0
    
    # Parse start date
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    
    # Update
    updater = FeatureStoreUpdater(feature_store_path)
    updater.update_all_timeframes(args.symbol, start_date)
    
    return 0


if __name__ == '__main__':
    exit(main())

