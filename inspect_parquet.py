#!/usr/bin/env python3
"""
Quick script to inspect parquet file contents.
"""
import pandas as pd
from pathlib import Path
import sys

def inspect_parquet(file_path: Path):
    """Inspect a parquet file and print its contents."""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print("="*80)
    print(f"Inspecting: {file_path.name}")
    print("="*80)
    
    # Load parquet file
    df = pd.read_parquet(file_path)
    
    print(f"\n📊 Data Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Check date range - handle both column and index
    timestamp_col = None
    if 'timestamp' in df.columns:
        timestamp_col = df['timestamp']
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamp_col = df.index
    elif df.index.name == 'timestamp' or (hasattr(df.index, 'name') and 'timestamp' in str(df.index.name).lower()):
        timestamp_col = df.index
    
    if timestamp_col is not None:
        if not isinstance(timestamp_col, pd.DatetimeIndex):
            timestamp_col = pd.to_datetime(timestamp_col)
        
        print(f"\n📅 Date Range:")
        print(f"   From: {timestamp_col.min()}")
        print(f"   To:   {timestamp_col.max()}")
        years = pd.Series(timestamp_col).dt.year.unique()
        print(f"   Years: {sorted(years)}")
        print(f"   Total years: {len(years)} ({min(years)} to {max(years)})")
        print(f"   Date span: {(timestamp_col.max() - timestamp_col.min()).days} days")
    else:
        print(f"\n⚠️  No timestamp found in columns or index")
    
    # List all columns
    print(f"\n📋 All Columns ({len(df.columns)} total):")
    all_cols = sorted(df.columns)
    
    # Categorize columns
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'atr']
    quote_cols = []
    indicator_cols = []
    engineered_cols = []
    
    for col in all_cols:
        col_lower = col.lower()
        if col in ohlcv_cols:
            continue
        elif any(kw in col_lower for kw in ['bid', 'ask', 'spread', 'quote', 'mid', 'depth', 'volume_bid', 'volume_ask']):
            quote_cols.append(col)
        elif any(kw in col_lower for kw in ['regime', 'momentum', 'mr_', 'micro_', 'liq_sweep']):
            engineered_cols.append(col)
        else:
            indicator_cols.append(col)
    
    print(f"\n   OHLCV Columns ({len([c for c in ohlcv_cols if c in all_cols])}):")
    for col in ohlcv_cols:
        if col in all_cols:
            non_null = df[col].notna().sum()
            print(f"      • {col} ({non_null:,} non-null)")
    
    if quote_cols:
        print(f"\n   💰 Quote Features ({len(quote_cols)}):")
        for col in sorted(quote_cols):
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100 if len(df) > 0 else 0
            print(f"      • {col} ({non_null:,} non-null, {pct:.1f}%)")
    else:
        print(f"\n   💰 Quote Features: None found")
    
    if indicator_cols:
        print(f"\n   📈 Indicator Features ({len(indicator_cols)}):")
        for col in sorted(indicator_cols)[:20]:  # Show first 20
            non_null = df[col].notna().sum()
            print(f"      • {col} ({non_null:,} non-null)")
        if len(indicator_cols) > 20:
            print(f"      ... and {len(indicator_cols) - 20} more indicator columns")
    
    if engineered_cols:
        print(f"\n   🔧 Engineered Features ({len(engineered_cols)}):")
        for col in sorted(engineered_cols):
            non_null = df[col].notna().sum()
            print(f"      • {col} ({non_null:,} non-null)")
    
    # Show sample
    print(f"\n📄 Sample Data (first 2 rows, showing first 15 columns):")
    sample_cols = list(df.columns[:15])
    print(df[sample_cols].head(2).to_string())
    
    # Show quote features sample
    if quote_cols:
        print(f"\n💰 Quote Features Sample (first 2 rows):")
        print(df[quote_cols[:10]].head(2).to_string())
    
    print("\n" + "="*80)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Default: check XAUUSD 5T
        file_path = Path("ML_model/ML_model/feature_store/XAUUSD/XAUUSD_5T.parquet")
    
    inspect_parquet(file_path)
    
    # Also check quotes directory if it exists
    quotes_dir = Path("ML_model/ML_model/feature_store/quotes/XAUUSD")
    if quotes_dir.exists():
        print("\n" + "="*80)
        print("Also checking quotes directory...")
        print("="*80)
        for quote_file in quotes_dir.glob("*.parquet"):
            print(f"\n📁 Quotes file: {quote_file.name}")
            inspect_parquet(quote_file)

