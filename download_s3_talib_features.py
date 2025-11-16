#!/usr/bin/env python3
"""
Download historical forex data from Massive.com S3 and calculate features using TA-Lib.

Process:
1. Download raw minute OHLCV from Massive.com S3
2. Resample to target timeframes
3. Calculate technical indicators using TA-Lib
4. Save to feature_store parquet files
"""

import os
import sys
import boto3
import pandas as pd
import numpy as np
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

try:
    import talib
except ImportError:
    print("❌ TA-Lib not installed. Install with: pip install TA-Lib")
    sys.exit(1)

load_dotenv()

# S3 Configuration (Massive.com)
ACCESS_KEY_ID = os.getenv('Access_Key_ID')
SECRET_ACCESS_KEY = os.getenv('Secret_Access_Key')
S3_ENDPOINT = os.getenv('S3_endpoint')
BUCKET = os.getenv('Bucket')

if not all([ACCESS_KEY_ID, SECRET_ACCESS_KEY, S3_ENDPOINT, BUCKET]):
    print("❌ Missing S3 credentials in .env file!")
    print("Required: Access_Key_ID, Secret_Access_Key, S3_endpoint, Bucket")
    sys.exit(1)

# Symbols to download (Polygon format in S3 - with hyphens!)
SYMBOLS = {
    'XAUUSD': 'C:XAU-USD',
    'XAGUSD': 'C:XAG-USD',
}

TIMEFRAMES = ['5T', '15T', '30T', '1H']

# Paths
FEATURE_STORE_DIR = Path('feature_store')


def setup_s3_client():
    """Initialize S3 client for Massive.com."""
    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
    )
    return session.client('s3', endpoint_url=S3_ENDPOINT)


def download_day(s3, date_str, polygon_ticker):
    """
    Download one day of minute data from S3.

    Returns:
        DataFrame with OHLCV minute data, or None if not found
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    # Skip weekends
    if date_obj.weekday() >= 5:
        return None

    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')

    # S3 key for Massive.com Polygon data
    file_key = f'global_forex/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'

    try:
        # Fetch from S3
        response = s3.get_object(Bucket=BUCKET, Key=file_key)
        compressed_data = response['Body'].read()

        # Decompress and parse
        with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
            df = pd.read_csv(gz)

        # Filter for our ticker
        if 'ticker' in df.columns:
            df = df[df['ticker'] == polygon_ticker].copy()
        else:
            return None

        if df.empty:
            return None

        # Convert timestamp (window_start is in nanoseconds)
        if 'window_start' in df.columns:
            df['timestamp'] = pd.to_datetime(df['window_start'], unit='ns', utc=True)
        elif 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'], unit='ns', utc=True)
        else:
            return None

        # Standardize columns
        column_map = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        }

        df = df.rename(columns=column_map)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.set_index('timestamp').sort_index()

        return df

    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        return None


def calculate_talib_features(df):
    """
    Calculate technical indicators using TA-Lib.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with all features
    """
    if df.empty or len(df) < 200:
        return None

    result = df.copy()

    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    open_ = df['open'].values.astype(float)
    volume = df['volume'].values.astype(float)

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

    # Candlestick patterns
    result['cdl_doji'] = talib.CDLDOJI(open_, high, low, close)
    result['cdl_hammer'] = talib.CDLHAMMER(open_, high, low, close)
    result['cdl_engulfing'] = talib.CDLENGULFING(open_, high, low, close)
    result['cdl_morning_star'] = talib.CDLMORNINGSTAR(open_, high, low, close)
    result['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(open_, high, low, close)

    # Price changes
    result['pct_change'] = close / np.roll(close, 1) - 1
    result['high_low_range'] = (high - low) / close

    # Drop NaN rows
    result = result.dropna()

    return result


def download_and_process_symbol(symbol, start_date, end_date):
    """
    Download all data for one symbol and create feature parquet files.
    """
    polygon_ticker = SYMBOLS[symbol]

    print(f"\n{'='*80}")
    print(f"Processing {symbol} ({polygon_ticker})")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*80}\n")

    s3 = setup_s3_client()

    # Calculate total days
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - current).days + 1

    # Download all minute data
    print("📥 Downloading minute data from Massive.com S3...")
    print(f"   Total days to process: {total_days}")
    print()

    all_minute_bars = []
    day_count = 0
    success_count = 0

    # Progress tracking
    last_update = datetime.now()
    start_time = datetime.now()

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        df_day = download_day(s3, date_str, polygon_ticker)
        if df_day is not None:
            all_minute_bars.append(df_day)
            success_count += 1

        day_count += 1

        # Update progress every second
        now = datetime.now()
        if (now - last_update).total_seconds() >= 1.0 or day_count % 50 == 0:
            progress_pct = (day_count / total_days) * 100
            elapsed = (now - start_time).total_seconds()

            # Estimate time remaining
            if day_count > 0:
                rate = day_count / elapsed
                remaining_days = total_days - day_count
                eta_seconds = remaining_days / rate if rate > 0 else 0
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "calculating..."

            # Progress bar
            bar_length = 40
            filled = int(bar_length * day_count / total_days)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"\r  [{bar}] {progress_pct:.1f}% | {day_count}/{total_days} days | {success_count} successful | ETA: {eta_str}  ", end='', flush=True)
            last_update = now

        current += timedelta(days=1)

    # Final update
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  ✓ Downloaded {success_count}/{day_count} days in {int(elapsed)}s")

    if not all_minute_bars:
        print(f"  ❌ No data found for {symbol}")
        return

    # Combine all minute data
    minute_df = pd.concat(all_minute_bars).sort_index()
    minute_df = minute_df[~minute_df.index.duplicated(keep='last')]

    print(f"  📊 Total minute bars: {len(minute_df):,}")
    print(f"  📅 Range: {minute_df.index.min()} to {minute_df.index.max()}")

    # Process each timeframe
    for i, timeframe in enumerate(TIMEFRAMES, 1):
        print(f"\n🔧 Processing {timeframe} ({i}/{len(TIMEFRAMES)})...")

        # Resample
        if timeframe == '1H':
            resample_rule = '60T'
        else:
            resample_rule = timeframe

        print(f"  ⏳ Resampling to {timeframe}...", end='', flush=True)
        df_tf = minute_df.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        print(f"\r  ✓ Resampled: {len(df_tf):,} bars")

        # Calculate features using TA-Lib
        print(f"  ⏳ Calculating TA-Lib features...", end='', flush=True)
        df_features = calculate_talib_features(df_tf)

        if df_features is None or df_features.empty:
            print(f"\r  ❌ Feature calculation failed")
            continue

        feature_count = len(df_features.columns)
        print(f"\r  ✓ Calculated {feature_count} TA-Lib features for {len(df_features):,} bars")

        # Save to parquet
        output_dir = FEATURE_STORE_DIR / symbol
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{symbol}_{timeframe}.parquet"

        print(f"  ⏳ Saving to parquet...", end='', flush=True)
        df_features.to_parquet(output_file, compression='snappy')

        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"\r  ✓ Saved: {output_file.name} ({file_size_mb:.1f} MB)")

    print(f"\n✅ {symbol} complete!\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download historical data and calculate TA-Lib features')
    parser.add_argument('--start-date', default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', choices=list(SYMBOLS.keys()) + ['all'], default='all', help='Symbol to process')

    args = parser.parse_args()

    # Determine which symbols to process
    if args.symbol == 'all':
        symbols_to_process = list(SYMBOLS.keys())
    else:
        symbols_to_process = [args.symbol]

    print("\n" + "="*80)
    print("HISTORICAL DATA DOWNLOAD & TA-LIB FEATURE GENERATION")
    print("="*80)
    print(f"Source: Massive.com S3 (Polygon data)")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Symbols: {', '.join(symbols_to_process)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Features: TA-Lib technical indicators")
    print("="*80)

    # Process each symbol
    for symbol in symbols_to_process:
        download_and_process_symbol(symbol, args.start_date, args.end_date)

    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)
    print("\nFeature files saved to:")
    for symbol in symbols_to_process:
        print(f"  feature_store/{symbol}/")
    print()


if __name__ == "__main__":
    main()
