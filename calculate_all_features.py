#!/usr/bin/env python3
"""
Calculate all features from downloaded raw OHLCV data.
Reads from feature_store parquet files and enriches with 50+ technical indicators.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Optional

# Import feature engineering from institutional system
sys.path.insert(0, str(Path(__file__).parent))
from institutional_ml_trading_system import FeatureEngineer


def load_raw_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load raw OHLCV data from feature store."""
    file_path = Path("feature_store") / symbol / f"{symbol}_{timeframe}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_parquet(file_path)

    # Handle timestamp in index or column
    if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if df.columns[0] != 'timestamp':
            df = df.rename(columns={df.columns[0]: 'timestamp'})

    print(f"\n📊 Loaded {symbol} {timeframe}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    if 'timestamp' in df.columns:
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def save_features(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save feature-enriched data back to feature store."""
    output_dir = Path("feature_store") / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{symbol}_{timeframe}.parquet"

    # Save with compression
    df.to_parquet(output_file, index=False, compression='snappy')

    file_size_mb = output_file.stat().st_size / 1024 / 1024

    print(f"\n💾 Saved enriched data to {output_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Size: {file_size_mb:.2f} MB")

    return output_file


def main():
    """Calculate features for all symbols and timeframes."""

    SYMBOLS = ['XAUUSD', 'XAGUSD']
    TIMEFRAMES = ['5T', '15T', '30T', '1H']

    print("\n" + "="*80)
    print("FEATURE CALCULATION PIPELINE")
    print("="*80)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")

    # Load all data first
    data_cache = {}

    for symbol in SYMBOLS:
        data_cache[symbol] = {}
        for timeframe in TIMEFRAMES:
            try:
                df = load_raw_data(symbol, timeframe)
                data_cache[symbol][timeframe] = df
            except FileNotFoundError as e:
                print(f"⚠️  {e}")
                continue

    # Calculate features for each symbol/timeframe
    for symbol in SYMBOLS:
        print(f"\n{'#'*80}")
        print(f"# CALCULATING FEATURES FOR {symbol}")
        print(f"{'#'*80}")

        for timeframe in TIMEFRAMES:
            if timeframe not in data_cache[symbol]:
                continue

            print(f"\n{'='*80}")
            print(f"Processing {symbol} {timeframe}")
            print(f"{'='*80}")

            df = data_cache[symbol][timeframe].copy()

            # Determine secondary symbol for cross-asset features
            secondary_symbol = 'XAGUSD' if symbol == 'XAUUSD' else 'XAUUSD'
            df_secondary = data_cache.get(secondary_symbol, {}).get(timeframe)

            # Calculate all features
            try:
                df_features = FeatureEngineer.create_all_features(
                    df=df,
                    df_secondary=df_secondary,
                    primary_symbol=symbol,
                    secondary_symbol=secondary_symbol
                )

                print(f"\n✅ Feature engineering complete!")
                print(f"   Input rows: {len(df):,}")
                print(f"   Output rows: {len(df_features):,}")
                print(f"   Features created: {len(df_features.columns) - 6}")  # Subtract OHLCV+timestamp

                # Show sample features
                feature_cols = [col for col in df_features.columns
                               if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

                print(f"\n   Sample features ({len(feature_cols)} total):")
                for feat in feature_cols[:10]:
                    values = df_features[feat].dropna()
                    if len(values) > 0:
                        print(f"     {feat:30s}: mean={values.mean():8.4f}, std={values.std():8.4f}")

                # Save enriched data
                save_features(df_features, symbol, timeframe)

            except Exception as e:
                print(f"\n❌ Error calculating features for {symbol} {timeframe}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "="*80)
    print("✅ FEATURE CALCULATION COMPLETE!")
    print("="*80)

    # Summary
    print("\n📊 Final Summary:")
    feature_store = Path("feature_store")

    for symbol in SYMBOLS:
        print(f"\n{symbol}:")
        symbol_dir = feature_store / symbol

        if symbol_dir.exists():
            for timeframe in TIMEFRAMES:
                file_path = symbol_dir / f"{symbol}_{timeframe}.parquet"

                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    feature_count = len([col for col in df.columns
                                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])

                    print(f"  {timeframe:5s}: {len(df):7,} rows | {feature_count:3d} features | {size_mb:6.2f} MB")

    print("\n🚀 Ready to train institutional ML system!")
    print("   Next: python3 institutional_ml_trading_system.py")


if __name__ == "__main__":
    main()
