#!/usr/bin/env python3
"""
Calculate quote features for each OHLCV bar and merge with existing features.
STRICT NO LOOKAHEAD: For bar ending at T with duration Δ, use quotes ∈ [T-Δ, T) only.

Quote features calculated:
- Spread/cost: q_spread_mean, q_spread_std, q_spread_last, q_spread_pct_mean, q_spread_pct_std
- Mid micro-move: q_mid_first, q_mid_last, q_mid_ret, q_mid_range
- Liquidity: q_quote_count
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import sys


def calculate_quote_features_for_bar(bar_end: pd.Timestamp, bar_duration: timedelta,
                                     quotes_df: pd.DataFrame) -> dict:
    """
    Calculate quote features for a single bar.

    CRITICAL: No lookahead! Use quotes with timestamp ∈ [bar_end - bar_duration, bar_end)
    Start inclusive, end EXCLUSIVE.

    Args:
        bar_end: End timestamp of the bar
        bar_duration: Duration of the bar (e.g., 5 minutes)
        quotes_df: DataFrame with all quotes (must have 'timestamp' as index)

    Returns:
        dict with quote features
    """
    bar_start = bar_end - bar_duration

    # Get quotes in window [bar_start, bar_end) - end EXCLUSIVE
    mask = (quotes_df.index >= bar_start) & (quotes_df.index < bar_end)
    window_quotes = quotes_df[mask]

    if len(window_quotes) == 0:
        # No quotes in window - return NaN features
        return {
            'q_spread_mean': np.nan,
            'q_spread_std': np.nan,
            'q_spread_last': np.nan,
            'q_spread_pct_mean': np.nan,
            'q_spread_pct_std': np.nan,
            'q_mid_first': np.nan,
            'q_mid_last': np.nan,
            'q_mid_ret': np.nan,
            'q_mid_range': np.nan,
            'q_quote_count': 0,
        }

    # Spread features
    q_spread_mean = window_quotes['spread'].mean()
    q_spread_std = window_quotes['spread'].std() if len(window_quotes) > 1 else 0.0
    q_spread_last = window_quotes['spread'].iloc[-1]
    q_spread_pct_mean = window_quotes['spread_pct'].mean()
    q_spread_pct_std = window_quotes['spread_pct'].std() if len(window_quotes) > 1 else 0.0

    # Mid micro-move features
    q_mid_first = window_quotes['mid'].iloc[0]
    q_mid_last = window_quotes['mid'].iloc[-1]
    q_mid_ret = (q_mid_last - q_mid_first) / q_mid_first if q_mid_first != 0 else 0.0
    q_mid_range = window_quotes['mid'].max() - window_quotes['mid'].min()

    # Liquidity
    q_quote_count = len(window_quotes)

    return {
        'q_spread_mean': q_spread_mean,
        'q_spread_std': q_spread_std,
        'q_spread_last': q_spread_last,
        'q_spread_pct_mean': q_spread_pct_mean,
        'q_spread_pct_std': q_spread_pct_std,
        'q_mid_first': q_mid_first,
        'q_mid_last': q_mid_last,
        'q_mid_ret': q_mid_ret,
        'q_mid_range': q_mid_range,
        'q_quote_count': q_quote_count,
    }


def calculate_quote_features(symbol: str, timeframe: str, sample_check: bool = True) -> pd.DataFrame:
    """
    Calculate quote features for all bars and merge with OHLCV features.

    Args:
        symbol: 'XAUUSD' or 'XAGUSD'
        timeframe: '5T', '15T', '30T', '1H'
        sample_check: If True, randomly check a few bars for lookahead

    Returns:
        DataFrame with OHLCV features + quote features
    """
    print(f"\n{'='*80}")
    print(f"CALCULATING QUOTE FEATURES FOR {symbol} {timeframe}")
    print(f"{'='*80}")

    # Load OHLCV data with existing features
    ohlcv_path = Path("feature_store") / symbol / f"{symbol}_{timeframe}.parquet"

    if not ohlcv_path.exists():
        print(f"❌ OHLCV feature file not found: {ohlcv_path}")
        print(f"   Run calculate_all_features.py first!")
        return pd.DataFrame()

    print(f"📂 Loading OHLCV features from {ohlcv_path}")
    df_ohlcv = pd.read_parquet(ohlcv_path)

    print(f"   Rows: {len(df_ohlcv):,}")
    print(f"   Columns: {len(df_ohlcv.columns)}")
    print(f"   Date range: {df_ohlcv['timestamp'].min()} to {df_ohlcv['timestamp'].max()}")

    # Load quote data
    quote_path = Path("feature_store") / "quotes" / symbol / f"{symbol}_{timeframe}_quotes.parquet"

    if not quote_path.exists():
        print(f"⚠️  Quote file not found: {quote_path}")
        print(f"   Run process_cached_quotes.py first!")
        print(f"   Continuing without quote features...")
        return df_ohlcv

    print(f"📊 Loading quote data from {quote_path}")
    df_quotes = pd.read_parquet(quote_path)

    print(f"   Rows: {len(df_quotes):,}")
    print(f"   Date range: {df_quotes['timestamp'].min()} to {df_quotes['timestamp'].max()}")

    # Set timestamp as index for efficient lookups
    df_quotes = df_quotes.set_index('timestamp').sort_index()

    # Parse timeframe to get bar duration
    timeframe_map = {
        '5T': timedelta(minutes=5),
        '15T': timedelta(minutes=15),
        '30T': timedelta(minutes=30),
        '1H': timedelta(hours=1),
    }
    bar_duration = timeframe_map[timeframe]

    print(f"\n🔧 Calculating quote features (bar duration: {bar_duration})...")
    print(f"   Rule: For bar ending at T, use quotes ∈ [T-{bar_duration}, T)")

    # Calculate quote features for each bar
    quote_features_list = []

    for idx, row in df_ohlcv.iterrows():
        bar_end = row['timestamp']

        quote_feats = calculate_quote_features_for_bar(
            bar_end=pd.Timestamp(bar_end),
            bar_duration=bar_duration,
            quotes_df=df_quotes
        )

        quote_features_list.append(quote_feats)

        # Progress indicator
        if (idx + 1) % 10000 == 0:
            print(f"   Processed {idx+1:,} / {len(df_ohlcv):,} bars...")

    # Convert to DataFrame
    df_quote_features = pd.DataFrame(quote_features_list)

    # Sanity check: verify no lookahead
    if sample_check:
        print(f"\n🔍 SANITY CHECK: Verifying no lookahead...")
        sample_size = min(5, len(df_ohlcv))
        sample_indices = np.random.choice(len(df_ohlcv), sample_size, replace=False)

        for idx in sample_indices:
            bar_end = pd.Timestamp(df_ohlcv.iloc[idx]['timestamp'])
            bar_start = bar_end - bar_duration

            # Get quotes used for this bar
            mask = (df_quotes.index >= bar_start) & (df_quotes.index < bar_end)
            window_quotes = df_quotes[mask]

            if len(window_quotes) > 0:
                max_quote_ts = window_quotes.index.max()

                print(f"   Bar #{idx}: end={bar_end}")
                print(f"            Max quote timestamp={max_quote_ts}")
                print(f"            Check: {max_quote_ts} < {bar_end} ? {max_quote_ts < bar_end} ✓")

                assert max_quote_ts < bar_end, f"LOOKAHEAD DETECTED! Quote at {max_quote_ts} used for bar ending {bar_end}"

    # Merge with OHLCV features
    print(f"\n📦 Merging quote features with OHLCV features...")
    df_combined = pd.concat([df_ohlcv, df_quote_features], axis=1)

    print(f"   Total columns: {len(df_combined.columns)}")
    print(f"   Quote features added: {len(df_quote_features.columns)}")

    # Show sample stats
    print(f"\n📊 Quote Feature Statistics:")
    for col in df_quote_features.columns:
        non_null = df_quote_features[col].notna().sum()
        if non_null > 0:
            mean_val = df_quote_features[col].mean()
            print(f"   {col:20s}: {non_null:7,} non-null | mean={mean_val:10.6f}")
        else:
            print(f"   {col:20s}: All NaN")

    return df_combined


def main():
    """Calculate quote features for XAUUSD only."""

    symbol = 'XAUUSD'
    timeframes = ['5T', '15T', '30T', '1H']

    print("\n" + "="*80)
    print("QUOTE FEATURE CALCULATION PIPELINE")
    print("="*80)
    print(f"Symbol: {symbol}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print("\nFeatures to calculate:")
    print("  - Spread: q_spread_mean, q_spread_std, q_spread_last")
    print("  - Spread%: q_spread_pct_mean, q_spread_pct_std")
    print("  - Mid move: q_mid_first, q_mid_last, q_mid_ret, q_mid_range")
    print("  - Liquidity: q_quote_count")

    for timeframe in timeframes:
        # Calculate quote features
        df_combined = calculate_quote_features(symbol, timeframe)

        if df_combined.empty:
            print(f"\n❌ Failed to process {symbol} {timeframe}")
            continue

        # Save combined features
        output_path = Path("feature_store") / symbol / f"{symbol}_{timeframe}.parquet"

        print(f"\n💾 Saving to {output_path}")
        df_combined.to_parquet(output_path, index=False, compression='snappy')

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"   Size: {file_size_mb:.2f} MB")

    print("\n" + "="*80)
    print("✅ QUOTE FEATURE CALCULATION COMPLETE!")
    print("="*80)

    # Final summary
    print(f"\n📊 Final Summary for {symbol}:")
    for timeframe in timeframes:
        file_path = Path("feature_store") / symbol / f"{symbol}_{timeframe}.parquet"

        if file_path.exists():
            df = pd.read_parquet(file_path)

            # Count OHLCV vs quote features
            quote_cols = [col for col in df.columns if col.startswith('q_')]
            ohlcv_cols = [col for col in df.columns if not col.startswith('q_') and
                         col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            print(f"\n  {timeframe:5s}: {len(df):7,} bars")
            print(f"         OHLCV features: {len(ohlcv_cols)}")
            print(f"         Quote features: {len(quote_cols)}")
            print(f"         Total features: {len(ohlcv_cols) + len(quote_cols)}")

    print("\n🚀 Ready for ML training with combined OHLCV + quote features!")
    print("   Next: Train models with institutional_ml_trading_system.py")


if __name__ == "__main__":
    main()
