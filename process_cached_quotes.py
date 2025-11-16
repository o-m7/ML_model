#!/usr/bin/env python3
"""
Process existing cached quote data without downloading more.
Uses whatever months are already in quotes_cache/ and creates final quote files.
"""

import pandas as pd
from pathlib import Path

SYMBOLS = ['XAUUSD', 'XAGUSD']
TIMEFRAMES = ['5T', '15T', '30T', '1H']


def resample_quotes(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample quote data to specified timeframe."""
    if df.empty:
        return df

    df = df.set_index('timestamp').sort_index()

    # Resample quotes - use mean for spread metrics
    agg_dict = {
        'bid': 'mean',
        'ask': 'mean',
        'spread': 'mean',
        'spread_pct': 'mean',
        'mid': 'mean',
    }

    if 'bid_size' in df.columns:
        agg_dict['bid_size'] = 'sum'
        agg_dict['ask_size'] = 'sum'

    resampled = df.resample(timeframe).agg(agg_dict).dropna()
    resampled = resampled.reset_index()

    return resampled


def save_quotes(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save quote data to feature store."""
    output_dir = Path("feature_store") / "quotes" / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{symbol}_{timeframe}_quotes.parquet"

    df.to_parquet(output_file, index=False, compression='snappy')

    file_size_mb = output_file.stat().st_size / 1024 / 1024

    print(f"\n💾 Saved to {output_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Size: {file_size_mb:.2f} MB")


def main():
    """Process cached quote data."""

    print("\n" + "="*80)
    print("PROCESSING EXISTING CACHED QUOTE DATA")
    print("="*80)

    cache_base = Path("feature_store") / "quotes_cache"

    for symbol in SYMBOLS:
        print(f"\n{'#'*80}")
        print(f"# PROCESSING {symbol}")
        print(f"{'#'*80}")

        cache_dir = cache_base / symbol

        if not cache_dir.exists():
            print(f"⚠️  No cache found for {symbol} at {cache_dir}")
            continue

        # Find all cached month files
        cached_files = sorted(cache_dir.glob(f"{symbol}_*_*_raw.parquet"))

        if not cached_files:
            print(f"⚠️  No cached files found for {symbol}")
            continue

        print(f"\n📦 Found {len(cached_files)} cached months:")
        for f in cached_files[:5]:
            print(f"   - {f.name}")
        if len(cached_files) > 5:
            print(f"   ... and {len(cached_files) - 5} more")

        # Load all cached months
        print(f"\n📥 Loading cached data...")
        monthly_dfs = []
        for cache_file in cached_files:
            df_month = pd.read_parquet(cache_file)
            monthly_dfs.append(df_month)

        # Combine all months
        print(f"📊 Combining {len(monthly_dfs)} months...")
        df_quotes = pd.concat(monthly_dfs, ignore_index=True)
        df_quotes = df_quotes.sort_values('timestamp')
        df_quotes = df_quotes.drop_duplicates(subset='timestamp', keep='last')

        print(f"\n✅ Loaded {len(df_quotes):,} quote records")
        print(f"   Date range: {df_quotes['timestamp'].min()} to {df_quotes['timestamp'].max()}")
        print(f"   Avg spread: {df_quotes['spread'].mean():.4f}")
        print(f"   Avg spread %: {df_quotes['spread_pct'].mean():.4f}%")

        # Resample to all timeframes
        for timeframe in TIMEFRAMES:
            print(f"\n📊 Resampling quotes to {timeframe}...")

            df_resampled = resample_quotes(df_quotes.copy(), timeframe)

            if not df_resampled.empty:
                print(f"   ✅ {len(df_resampled):,} bars")
                save_quotes(df_resampled, symbol, timeframe)
            else:
                print(f"   ⚠️  No data after resampling")

    print("\n" + "="*80)
    print("✅ QUOTE PROCESSING COMPLETE!")
    print("="*80)

    # Summary
    print("\n📊 Final Summary:")
    quotes_dir = Path("feature_store") / "quotes"

    for symbol in SYMBOLS:
        print(f"\n{symbol}:")
        symbol_dir = quotes_dir / symbol

        if symbol_dir.exists():
            for timeframe in TIMEFRAMES:
                file_path = symbol_dir / f"{symbol}_{timeframe}_quotes.parquet"

                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
                    avg_spread = df['spread_pct'].mean()
                    print(f"  {timeframe:5s}: {len(df):7,} bars | {date_range:4d} days | Avg spread: {avg_spread:.4f}% | {size_mb:6.2f} MB")

    print("\n🚀 Ready to proceed with ML training!")
    print("   Next steps:")
    print("   1. python3 calculate_all_features.py")
    print("   2. python3 institutional_ml_trading_system.py")


if __name__ == "__main__":
    main()
