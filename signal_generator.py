#!/usr/bin/env python3
"""
STANDALONE SIGNAL GENERATOR FOR GITHUB ACTIONS
================================================
Generates signals without needing the API server running.
Uses ensemble predictions from multiple models.
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
import boto3
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client
import pandas_ta as ta

# Import ensemble predictor and news filter
sys.path.insert(0, str(Path(__file__).parent))
from ensemble_predictor import EnsemblePredictor
from news_filter import is_in_blackout_window
from production_final_system import BalancedModel
from live_feature_utils import build_feature_frame

# Ensure BalancedModel is available for pickle when this script runs as __main__
_main_module = sys.modules.get("__main__")
if _main_module is not None and not hasattr(_main_module, "BalancedModel"):
    setattr(_main_module, "BalancedModel", BalancedModel)

# Load environment
load_dotenv()

# Configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
POLYGON_S3_ACCESS_KEY = os.getenv('POLYGON_S3_ACCESS_KEY')
POLYGON_S3_SECRET_KEY = os.getenv('POLYGON_S3_SECRET_KEY')
POLYGON_S3_ENDPOINT = os.getenv('POLYGON_S3_ENDPOINT')
POLYGON_S3_BUCKET = os.getenv('POLYGON_S3_BUCKET')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY, POLYGON_S3_ACCESS_KEY, POLYGON_S3_SECRET_KEY]):
    print("❌ Missing required environment variables!")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize S3 client for Polygon flat files
s3_client = boto3.client(
    's3',
    aws_access_key_id=POLYGON_S3_ACCESS_KEY,
    aws_secret_access_key=POLYGON_S3_SECRET_KEY,
    endpoint_url=POLYGON_S3_ENDPOINT
)

# Models to process (from your production system)
MODELS = [
    ('AUDUSD', '15T'), ('AUDUSD', '30T'), ('AUDUSD', '5T'), ('AUDUSD', '1H'),
    ('EURUSD', '30T'), ('EURUSD', '5T'),
    ('GBPUSD', '15T'), ('GBPUSD', '1H'), ('GBPUSD', '30T'), ('GBPUSD', '5T'),
    ('NZDUSD', '15T'), ('NZDUSD', '1H'), ('NZDUSD', '30T'), ('NZDUSD', '4H'), ('NZDUSD', '5T'),
    ('XAGUSD', '15T'), ('XAGUSD', '1H'), ('XAGUSD', '30T'), ('XAGUSD', '4H'), ('XAGUSD', '5T'),
    ('XAUUSD', '15T'), ('XAUUSD', '1H'), ('XAUUSD', '30T'), ('XAUUSD', '4H'), ('XAUUSD', '5T'),
]

# Ticker mapping for Polygon S3 flat files (uses hyphenated format)
TICKER_MAP = {
    'XAUUSD': 'C:XAU-USD',
    'XAGUSD': 'C:XAG-USD',
    'EURUSD': 'C:EUR-USD',
    'GBPUSD': 'C:GBP-USD',
    'AUDUSD': 'C:AUD-USD',
    'NZDUSD': 'C:NZD-USD',
}

# TP/SL Parameters
SYMBOL_PARAMS = {
    'XAUUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.6, 'sl': 1.0}, '30T': {'tp': 2.0, 'sl': 1.0}, '1H': {'tp': 2.2, 'sl': 1.0}, '4H': {'tp': 2.5, 'sl': 1.0}},
    'XAGUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 2.0, 'sl': 1.0}, '1H': {'tp': 2.2, 'sl': 1.0}, '4H': {'tp': 2.5, 'sl': 1.0}},
    'EURUSD': {'5T': {'tp': 1.2, 'sl': 1.0}, '15T': {'tp': 1.4, 'sl': 1.0}, '30T': {'tp': 2.0, 'sl': 1.0}, '1H': {'tp': 2.2, 'sl': 1.0}, '4H': {'tp': 2.5, 'sl': 1.0}},
    'GBPUSD': {'5T': {'tp': 1.5, 'sl': 1.0}, '15T': {'tp': 1.6, 'sl': 1.0}, '30T': {'tp': 2.0, 'sl': 1.0}, '1H': {'tp': 2.2, 'sl': 1.0}, '4H': {'tp': 2.5, 'sl': 1.0}},
    'AUDUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 2.0, 'sl': 1.0}, '1H': {'tp': 2.2, 'sl': 1.0}, '4H': {'tp': 2.5, 'sl': 1.0}},
    'NZDUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 2.0, 'sl': 1.0}, '1H': {'tp': 2.2, 'sl': 1.0}, '4H': {'tp': 2.5, 'sl': 1.0}},
}

TIMEFRAME_MINUTES = {'5T': 5, '15T': 15, '30T': 30, '1H': 60, '4H': 240}

# Number of bars to keep per timeframe (balances history with API limits)
BARS_PER_TF = {
    '5T': 400,
    '15T': 240,
    '30T': 160,
    '1H': 120,
    '4H': 80,
}

# Minimum bars needed per timeframe
MIN_BARS_REQUIRED = {
    '5T': 120,
    '15T': 120,
    '30T': 120,
    '1H': 80,
    '4H': 40,
}

# Cache for ensemble predictors (one per symbol)
ENSEMBLE_CACHE = {}

# Sentiment filtering thresholds
SENTIMENT_LONG_THRESHOLD = 0.2   # Only take LONG if sentiment > 0.2
SENTIMENT_SHORT_THRESHOLD = -0.2  # Only take SHORT if sentiment < -0.2


def get_ensemble(symbol: str) -> Optional[EnsemblePredictor]:
    """Get or create ensemble predictor for a symbol."""
    if symbol not in ENSEMBLE_CACHE:
        try:
            ENSEMBLE_CACHE[symbol] = EnsemblePredictor(symbol)
        except Exception as e:
            print(f"  ⚠️  Cannot load ensemble for {symbol}: {e}")
            return None
    return ENSEMBLE_CACHE[symbol]


def get_recent_sentiment(symbol: str) -> float:
    """Get most recent sentiment for symbol from Supabase."""
    try:
        # Get sentiment from last 6 hours
        response = supabase.table('sentiment_data').select(
            'aggregate_sentiment'
        ).eq(
            'symbol', symbol
        ).order(
            'timestamp', desc=True
        ).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            sentiment = response.data[0]['aggregate_sentiment']
            return float(sentiment)
        else:
            return 0.0  # Neutral if no data
    
    except Exception as e:
        print(f"  ⚠️  Error fetching sentiment for {symbol}: {e}")
        return 0.0  # Neutral on error


def fetch_polygon_data(symbol: str, timeframe: str, bars: int = 200):
    """Fetch OHLCV data from Polygon S3 flat files."""
    ticker = TICKER_MAP.get(symbol, symbol)
    minutes = TIMEFRAME_MINUTES[timeframe]
    
    # Fetch last N days from S3
    days_to_fetch = max(10, (bars * minutes) // (24 * 60) + 5)  # Ensure enough days
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_to_fetch)
    
    all_bars = []
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day_str = current_date.strftime('%Y-%m-%d')
        s3_key = f'global_forex/minute_aggs_v1/{year}/{month}/{day_str}.csv.gz'
        
        try:
            response = s3_client.get_object(Bucket=POLYGON_S3_BUCKET, Key=s3_key)
            df_day = pd.read_csv(io.BytesIO(response['Body'].read()), compression='gzip')
            
            # Filter for symbol
            df_day = df_day[df_day['ticker'] == ticker]
            
            if not df_day.empty:
                # Convert timestamp
                df_day['timestamp'] = pd.to_datetime(df_day['window_start'], utc=True)
                df_day = df_day[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')
                all_bars.append(df_day)
                
        except s3_client.exceptions.NoSuchKey:
            pass  # File doesn't exist, skip
        except Exception:
            pass  # Other error, skip
        
        current_date += timedelta(days=1)
    
    if not all_bars:
        return None
    
    # Combine all days
    result = pd.concat(all_bars).sort_index()
    
    # Resample to target timeframe if needed
    if minutes > 1:
        result = result.resample(f'{minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    # Return last N bars
    return result.tail(bars)


def generate_simple_signal(df):
    """Generate a simple signal based on price action (fallback when ensemble unavailable)."""
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    current = df.iloc[-1]
    
    if current['close'] > current['sma_20'] > current['sma_50'] and current['rsi'] < 70:
        return 'long', 0.55, 0.10
    elif current['close'] < current['sma_20'] < current['sma_50'] and current['rsi'] > 30:
        return 'short', 0.55, 0.10
    else:
        return 'long' if current['close'] > current['sma_20'] else 'short', 0.45, 0.05


def process_symbol(symbol, timeframe):
    """Process one symbol/timeframe using ensemble predictions."""
    try:
        blackout_result = is_in_blackout_window(symbol)
        if blackout_result['is_blackout']:
            print(f"  🚫 {symbol} {timeframe}: BLACKOUT - {blackout_result['reason']}")
            return

        raw_df = fetch_polygon_data(symbol, timeframe, bars=BARS_PER_TF.get(timeframe, 200))
        min_bars = MIN_BARS_REQUIRED.get(timeframe, 120)
        if raw_df is None or len(raw_df) < min_bars:
            got = 0 if raw_df is None else len(raw_df)
            print(f"  ⚠️  {symbol} {timeframe}: Insufficient data (have {got} < {min_bars} bars)")
            return

        last_bar_time = raw_df.index[-1]
        staleness = datetime.now(timezone.utc) - last_bar_time
        max_allowed = timedelta(minutes=TIMEFRAME_MINUTES[timeframe] * 2)
        if staleness > max_allowed:
            print(f"  ⚠️  {symbol} {timeframe}: Stale data (last bar {last_bar_time} UTC, Δ {staleness})")
            return

        feature_df = build_feature_frame(raw_df)
        if feature_df.empty:
            print(f"  ⚠️  {symbol} {timeframe}: Unable to build feature set (insufficient history or NaNs)")
            return

        features_row = feature_df.tail(1)

        ensemble = get_ensemble(symbol)
        required_features = set()
        if ensemble:
            for model_info in ensemble.models.values():
                required_features.update(model_info['features'])
        missing_features = required_features - set(features_row.columns)
        if missing_features:
            print(f"  ⚠️  {symbol} {timeframe}: Missing model features {sorted(missing_features)[:10]}... using fallback")
            ensemble = None

        if ensemble:
            ensemble_result = ensemble.ensemble_predict(features_row, strategy='performance_weighted')
            signal_type = ensemble_result['signal']
            confidence = ensemble_result['confidence']
            edge = ensemble_result['edge']
            num_models = ensemble_result['num_models']

            if num_models == 0 or signal_type == 'flat' or confidence < 0.35:
                reason = "no model votes" if num_models == 0 else f"conf {confidence:.3f}"
                print(f"  ⚠️  {symbol} {timeframe}: Ensemble fallback triggered ({reason})")
                signal_type, confidence, edge = generate_simple_signal(raw_df.copy())
                source = 'fallback'
            else:
                print(f"  📊 {symbol} {timeframe}: Ensemble ({num_models} models) → {signal_type.upper()} (conf: {confidence:.3f}, edge: {edge:.3f})")
                source = 'ensemble'
        else:
            signal_type, confidence, edge = generate_simple_signal(raw_df.copy())
            print(f"  ⚠️  {symbol} {timeframe}: Using fallback signal → {signal_type.upper()}")
            source = 'fallback'

        sentiment = get_recent_sentiment(symbol)
        if abs(sentiment) > 1e-3:
            if signal_type == 'long' and sentiment < SENTIMENT_LONG_THRESHOLD:
                print(f"  🚫 {symbol} {timeframe}: LONG filtered by sentiment ({sentiment:.3f} < {SENTIMENT_LONG_THRESHOLD})")
                return
            if signal_type == 'short' and sentiment > SENTIMENT_SHORT_THRESHOLD:
                print(f"  🚫 {symbol} {timeframe}: SHORT filtered by sentiment ({sentiment:.3f} > {SENTIMENT_SHORT_THRESHOLD})")
                return
            print(f"  ✅ {symbol} {timeframe}: Sentiment OK ({sentiment:.3f})")

        atr = float(features_row['atr14'].iloc[-1])
        if pd.isna(atr) or atr == 0:
            atr = float(raw_df['close'].iloc[-1]) * 0.02

        entry_price = float(raw_df['close'].iloc[-1])
        params = SYMBOL_PARAMS.get(symbol, {}).get(timeframe, {'tp': 1.5, 'sl': 1.0})

        if signal_type == 'long':
            tp_price = entry_price + (atr * params['tp'])
            sl_price = entry_price - (atr * params['sl'])
        else:
            tp_price = entry_price - (atr * params['tp'])
            sl_price = entry_price + (atr * params['sl'])

        supabase_payload = {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': signal_type,
            'confidence': float(confidence),
            'edge': float(edge),
            'entry_price': entry_price,
            'take_profit': tp_price,
            'stop_loss': sl_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        supabase.table('live_signals').insert(supabase_payload).execute()

        print(f"  ✅ {symbol} {timeframe}: {signal_type.upper()} @ {entry_price:.5f} (TP: {tp_price:.5f}, SL: {sl_price:.5f})")
    except Exception as e:
        print(f"  ❌ Error processing {symbol} {timeframe}: {e}")


def main():
    """Main execution - generates signals for all production models."""
    print("\n" + "="*80)
    print(f"STANDALONE SIGNAL GENERATOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    success_count = 0
    for symbol, timeframe in MODELS:
        process_symbol(symbol, timeframe)
        success_count += 1
        time.sleep(0.5)
    
    print("\n" + "="*80)
    print(f"✅ Processed {success_count}/{len(MODELS)} models")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

