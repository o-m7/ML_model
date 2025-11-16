#!/usr/bin/env python3
"""
Download raw OHLCV data from Polygon API and calculate features using TA-Lib.
Simple, reliable approach that works with basic Polygon API key.

Process:
1. Download raw minute data from Polygon API
2. Resample to target timeframes
3. Calculate technical indicators using TA-Lib
4. Save to feature_store parquet files
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests
import time

try:
    import talib
except ImportError:
    print("❌ TA-Lib not installed. Install with: pip install TA-Lib")
    sys.exit(1)

load_dotenv()

# Polygon API configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY or POLYGON_API_KEY == 'your_api_key_here':
    print("❌ POLYGON_API_KEY not set in .env file!")
    print("Get your API key from: https://polygon.io/dashboard/api-keys")
    sys.exit(1)

POLYGON_BASE_URL = 'https://api.polygon.io/v2'

# Symbols to process
SYMBOLS = ['XAUUSD', 'XAGUSD']
TIMEFRAMES = ['5T', '15T', '30T', '1H']

# Paths
FEATURE_STORE_DIR = Path('feature_store')


def fetch_polygon_bars(symbol, multiplier, timespan, from_date, to_date):
    """
    Fetch aggregated bars from Polygon API.

    Args:
        symbol: e.g., 'C:XAUUSD'
        multiplier: e.g., 1
        timespan: 'minute', 'hour', 'day'
        from_date: 'YYYY-MM-DD'
        to_date: 'YYYY-MM-DD'

    Returns:
        DataFrame with OHLCV data
    """
    url = f"{POLYGON_BASE_URL}/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

    params = {
        'apiKey': POLYGON_API_KEY,
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000
    }

    all_results = []

    while url:
        try:
            response = requests.get(url, params=params if params else None)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'OK' and 'results' in data:
                all_results.extend(data['results'])

                # Check for next page
                if 'next_url' in data:
                    url = data['next_url']
                    params = None  # next_url includes all params
                    time.sleep(0.2)  # Rate limiting
                else:
                    break
            else:
                print(f"    ⚠️  No data: {data.get('status', 'Unknown error')}")
                break

        except Exception as e:
            print(f"    ❌ API Error: {e}")
            break

    if not all_results:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.set_index('timestamp').sort_index()

    return df


def calculate_talib_features(df):
    """
    Calculate technical indicators using TA-Lib.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)

    Returns:
        DataFrame with all features
    """
    if df.empty or len(df) < 200:
        return None

    result = df.copy()

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values

    print(f"    Calculating indicators...", end='', flush=True)

    # Trend Indicators
    result['sma_10'] = talib.SMA(close, timeperiod=10)
    result['sma_20'] = talib.SMA(close, timeperiod=20)
    result['sma_50'] = talib.SMA(close, timeperiod=50)
    result['sma_200'] = talib.SMA(close, timeperiod=200)
    result['ema_12'] = talib.EMA(close, timeperiod=12)
    result['ema_26'] = talib.EMA(close, timeperiod=26)

    # MACD
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    result['macd'] = macd
    result['macd_signal'] = macdsignal
    result['macd_hist'] = macdhist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    result['bb_upper'] = upper
    result['bb_middle'] = middle
    result['bb_lower'] = lower
    result['bb_width'] = (upper - lower) / middle

    # RSI
    result['rsi_14'] = talib.RSI(close, timeperiod=14)
    result['rsi_7'] = talib.RSI(close, timeperiod=7)

    # Stochastic
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    result['stoch_k'] = slowk
    result['stoch_d'] = slowd

    # ADX
    result['adx'] = talib.ADX(high, low, close, timeperiod=14)
    result['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    result['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

    # ATR
    result['atr'] = talib.ATR(high, low, close, timeperiod=14)
    result['natr'] = talib.NATR(high, low, close, timeperiod=14)

    # CCI
    result['cci'] = talib.CCI(high, low, close, timeperiod=14)

    # Williams %R
    result['willr'] = talib.WILLR(high, low, close, timeperiod=14)

    # MFI
    result['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)

    # OBV
    result['obv'] = talib.OBV(close, volume)

    # Momentum
    result['mom'] = talib.MOM(close, timeperiod=10)
    result['roc'] = talib.ROC(close, timeperiod=10)

    # Parabolic SAR
    result['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    # Average Price
    result['avgprice'] = talib.AVGPRICE(open_, high, low, close)
    result['medprice'] = talib.MEDPRICE(high, low)
    result['typprice'] = talib.TYPPRICE(high, low, close)
    result['wclprice'] = talib.WCLPRICE(high, low, close)

    # Candlestick patterns (returns -100, 0, or 100)
    result['cdl_doji'] = talib.CDLDOJI(open_, high, low, close)
    result['cdl_hammer'] = talib.CDLHAMMER(open_, high, low, close)
    result['cdl_engulfing'] = talib.CDLENGULFING(open_, high, low, close)
    result['cdl_morning_star'] = talib.CDLMORNINGSTAR(open_, high, low, close)
    result['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(open_, high, low, close)

    # Price changes
    result['pct_change'] = close / np.roll(close, 1) - 1
    result['high_low_range'] = (high - low) / close

    print(f"\r    ✓ Calculated {len(result.columns)} features")

    # Drop NaN rows (from indicator calculations)
    result = result.dropna()

    return result


def download_and_process(symbol, start_date, end_date):
    """
    Download raw data and create feature parquet files.
    """
    polygon_ticker = f'C:{symbol}'

    print(f"\n{'='*80}")
    print(f"Processing {symbol} ({polygon_ticker})")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*80}\n")

    # Download minute data
    print(f"📥 Downloading minute data from Polygon API...")
    df_minute = fetch_polygon_bars(
        polygon_ticker,
        multiplier=1,
        timespan='minute',
        from_date=start_date,
        to_date=end_date
    )

    if df_minute.empty:
        print(f"  ❌ No data fetched for {symbol}")
        return

    print(f"  ✓ Downloaded {len(df_minute):,} minute bars")
    print(f"  📅 Range: {df_minute.index.min()} to {df_minute.index.max()}")

    # Process each timeframe
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        print(f"\n🔧 Processing {timeframe} ({i}/{len(TIMEFRAMES)})...")

        # Resample
        if timeframe == '1H':
            resample_rule = '60T'
        else:
            resample_rule = timeframe

        print(f"    Resampling to {timeframe}...", end='', flush=True)
        df_tf = df_minute.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        print(f"\r    ✓ Resampled: {len(df_tf):,} bars")

        # Calculate features using TA-Lib
        df_features = calculate_talib_features(df_tf)

        if df_features is None or df_features.empty:
            print(f"    ❌ Feature calculation failed")
            continue

        # Save to parquet
        output_dir = FEATURE_STORE_DIR / symbol
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{symbol}_{timeframe}.parquet"

        print(f"    Saving to parquet...", end='', flush=True)
        df_features.to_parquet(output_file, compression='snappy')

        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"\r    ✓ Saved: {output_file.name} ({file_size_mb:.1f} MB, {len(df_features):,} bars)")

    print(f"\n✅ {symbol} complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download historical data and calculate TA-Lib features')
    parser.add_argument('--start-date', default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', choices=SYMBOLS + ['all'], default='all', help='Symbol to process')

    args = parser.parse_args()

    # Determine which symbols to process
    if args.symbol == 'all':
        symbols_to_process = SYMBOLS
    else:
        symbols_to_process = [args.symbol]

    print("\n" + "="*80)
    print("HISTORICAL DATA DOWNLOAD & TA-LIB FEATURE GENERATION")
    print("="*80)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Symbols: {', '.join(symbols_to_process)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Using: Polygon API + TA-Lib")
    print("="*80)

    # Process each symbol
    for symbol in symbols_to_process:
        download_and_process(symbol, args.start_date, args.end_date)

    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)
    print("\nFeature files saved to:")
    for symbol in symbols_to_process:
        print(f"  feature_store/{symbol}/")
    print()


if __name__ == "__main__":
    main()
