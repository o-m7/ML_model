#!/usr/bin/env python3
"""
CALCULATE COMPREHENSIVE FEATURES - OHLCV + QUOTES
==================================================

Merges OHLCV and quote data, calculates all technical indicators,
and saves complete feature sets as parquet files.

Usage:
    python calculate_all_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

# Configuration
SYMBOL = 'XAUUSD'
TIMEFRAMES = ['5T', '15T', '30T', '1H', '4H']

OHLCV_DIR = Path("feature_store") / SYMBOL
QUOTES_DIR = Path("feature_store") / "quotes" / SYMBOL
OUTPUT_DIR = Path("feature_store") / SYMBOL

print("\n" + "="*80)
print(f"CALCULATING FEATURES - {SYMBOL}")
print("="*80)


def load_data(timeframe: str):
    """Load OHLCV and quote data."""
    print(f"\n📁 Loading {timeframe} data...")

    # Load OHLCV
    ohlcv_path = OHLCV_DIR / f"{SYMBOL}_{timeframe}.parquet"
    if not ohlcv_path.exists():
        print(f"   ❌ OHLCV file not found: {ohlcv_path}")
        return None

    ohlcv_df = pd.read_parquet(ohlcv_path)
    print(f"   ✅ Loaded OHLCV: {len(ohlcv_df):,} bars")

    # Load quotes (optional)
    quotes_path = QUOTES_DIR / f"{SYMBOL}_{timeframe}_quotes.parquet"
    if quotes_path.exists():
        quotes_df = pd.read_parquet(quotes_path)
        print(f"   ✅ Loaded quotes: {len(quotes_df):,} bars")

        # Merge on timestamp
        df = pd.merge(ohlcv_df, quotes_df, on='timestamp', how='left', suffixes=('', '_quote'))
        print(f"   ✅ Merged: {len(df):,} bars")
    else:
        print(f"   ⚠️  Quote file not found, using OHLCV only")
        df = ohlcv_df

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical features."""
    print("\n🔧 Calculating features...")

    df = df.copy()

    # ATR (critical for labeling)
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(window=14).mean()
    df['atr20'] = tr.rolling(window=20).mean()

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages (multiple periods)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema{period}'] = df['close'].ewm(span=period).mean()
        df[f'close_vs_sma{period}'] = (df['close'] - df[f'sma{period}']) / (df[f'sma{period}'] + 1e-10)
        df[f'close_vs_ema{period}'] = (df['close'] - df[f'ema{period}']) / (df[f'ema{period}'] + 1e-10)

    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10)

    # RSI (multiple periods)
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi{period}'] = 100 - (100 / (1 + rs))

    # Bollinger Bands (multiple periods)
    for period in [10, 20, 30]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = sma + (std * 2)
        df[f'bb_lower_{period}'] = sma - (std * 2)
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                       (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / (sma + 1e-10)

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Volume features
    if 'volume' in df.columns:
        df['volume_sma10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma20'] + 1e-10)
        df['volume_std'] = df['volume'].rolling(window=20).std()

    # Price action
    df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    df['close_open_diff'] = (df['close'] - df['open']) / (df['open'] + 1e-10)

    # Candle patterns
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    df['body_size'] = body / (df['close'] + 1e-10)
    df['upper_shadow_ratio'] = upper_shadow / (body + 1e-10)
    df['lower_shadow_ratio'] = lower_shadow / (body + 1e-10)

    # ADX
    df['adx'] = calculate_adx(df, period=14)

    # Volatility
    for period in [10, 20, 50]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
        df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / \
                                             (df[f'volatility_{period}'].rolling(50).mean() + 1e-10)

    # Support/Resistance
    df['highest_high_20'] = df['high'].rolling(window=20).max()
    df['lowest_low_20'] = df['low'].rolling(window=20).min()
    df['dist_from_high'] = (df['highest_high_20'] - df['close']) / (df['atr14'] + 1e-10)
    df['dist_from_low'] = (df['close'] - df['lowest_low_20']) / (df['atr14'] + 1e-10)

    # Quote-specific features (if available)
    if 'spread' in df.columns:
        df['spread_pct'] = df['spread'] / (df['mid_close'] + 1e-10)
        df['spread_vs_atr'] = df['spread'] / (df['atr14'] + 1e-10)
        df['bid_ask_imbalance'] = (df['ask_close'] - df['bid_close']) / (df['ask_close'] + df['bid_close'] + 1e-10)

    if 'mid_close' in df.columns:
        df['close_vs_mid'] = (df['close'] - df['mid_close']) / (df['mid_close'] + 1e-10)

    # Drop NaN rows
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)

    print(f"   ✅ Calculated features, {len(df):,} bars (dropped {dropped:,} NaN rows)")
    print(f"   📊 Total features: {len(df.columns)}")

    return df


def calculate_adx(df, period=14):
    """Calculate Average Directional Index."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()

    return adx


def save_features(df: pd.DataFrame, timeframe: str):
    """Save feature DataFrame as parquet."""
    output_path = OUTPUT_DIR / f"{SYMBOL}_{timeframe}.parquet"

    print(f"\n💾 Saving features to {output_path}...")

    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved {len(df):,} rows, {len(df.columns)} columns ({file_size_mb:.2f} MB)")

    return output_path


def main():
    """Calculate features for all timeframes."""

    results = []

    for timeframe in TIMEFRAMES:
        print("\n" + "-"*80)
        print(f"PROCESSING: {timeframe}")
        print("-"*80)

        try:
            # Load data
            df = load_data(timeframe)
            if df is None or len(df) == 0:
                print(f"❌ Failed to load {timeframe} data")
                results.append({'timeframe': timeframe, 'success': False})
                continue

            # Calculate features
            df = calculate_features(df)

            # Save
            output_path = save_features(df, timeframe)

            results.append({
                'timeframe': timeframe,
                'success': True,
                'bars': len(df),
                'features': len(df.columns),
                'file': output_path
            })

        except Exception as e:
            print(f"❌ Error processing {timeframe}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'timeframe': timeframe, 'success': False})

    # Summary
    print("\n" + "="*80)
    print("FEATURE CALCULATION SUMMARY")
    print("="*80)

    for result in results:
        tf = result['timeframe']
        if result['success']:
            print(f"✅ {tf:6s}: {result['bars']:,} bars, {result['features']} features → {result['file'].name}")
        else:
            print(f"❌ {tf:6s}: Failed")

    print("="*80)

    successful = sum(1 for r in results if r['success'])
    print(f"\n✅ Successfully processed {successful}/{len(TIMEFRAMES)} timeframes")

    if successful > 0:
        print(f"\n📁 Features saved in: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("  1. Run: python train_all_timeframes_local.py")
        print("  2. Run: python validate_all_models.py --symbol XAUUSD")

    print("\n")

    return 0 if successful == len(TIMEFRAMES) else 1


if __name__ == '__main__':
    sys.exit(main())
