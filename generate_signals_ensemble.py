#!/usr/bin/env python3
"""
ENSEMBLE SIGNAL GENERATOR FOR GITHUB ACTIONS
==============================================
Generates signals using multi-model ensemble voting.
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import pandas_ta as ta

# Import ensemble predictor
sys.path.insert(0, str(Path(__file__).parent))
from ensemble_predictor import EnsemblePredictor

# Load environment
load_dotenv()

# Configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not all([POLYGON_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    print("❌ Missing required environment variables!")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Symbols to process (we'll use ensemble across timeframes for each symbol)
SYMBOLS = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']

# Ticker mapping for Polygon
TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD',
}

# TP/SL Parameters (use for final signal output)
SYMBOL_PARAMS = {
    'XAUUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 1.5, 'sl': 1.0}, '1H': {'tp': 1.6, 'sl': 1.0}, '4H': {'tp': 1.8, 'sl': 1.0}},
    'XAGUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 1.5, 'sl': 1.0}, '1H': {'tp': 1.5, 'sl': 1.0}, '4H': {'tp': 1.7, 'sl': 1.0}},
    'EURUSD': {'5T': {'tp': 1.2, 'sl': 1.0}, '15T': {'tp': 1.4, 'sl': 1.0}, '30T': {'tp': 1.3, 'sl': 1.0}, '1H': {'tp': 1.5, 'sl': 1.0}, '4H': {'tp': 1.6, 'sl': 1.0}},
    'GBPUSD': {'5T': {'tp': 1.5, 'sl': 1.0}, '15T': {'tp': 1.6, 'sl': 1.0}, '30T': {'tp': 1.6, 'sl': 1.0}, '1H': {'tp': 1.6, 'sl': 1.0}, '4H': {'tp': 1.7, 'sl': 1.0}},
    'AUDUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 1.5, 'sl': 1.0}, '1H': {'tp': 1.6, 'sl': 1.0}, '4H': {'tp': 1.7, 'sl': 1.0}},
    'NZDUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 1.5, 'sl': 1.0}, '1H': {'tp': 1.6, 'sl': 1.0}, '4H': {'tp': 1.7, 'sl': 1.0}},
}

TIMEFRAME_MINUTES = {'5T': 5, '15T': 15, '30T': 30, '1H': 60, '4H': 240}

# Ensemble strategy to use
ENSEMBLE_STRATEGY = 'performance_weighted'  # 'majority', 'confidence_weighted', or 'performance_weighted'


def fetch_polygon_data(symbol: str, timeframe: str, bars: int = 200):
    """Fetch OHLCV data from Polygon."""
    ticker = TICKER_MAP.get(symbol, symbol)
    minutes = TIMEFRAME_MINUTES[timeframe]
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=minutes * bars * 2)
    
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': POLYGON_API_KEY
    }
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{minutes}/minute/{int(start_time.timestamp()*1000)}/{int(end_time.timestamp()*1000)}"
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'results' not in data or not data['results']:
            return None
            
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')
        df = df.sort_index()
        
        return df.tail(bars)
    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return None


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features for the model.
    This should match the features used during training.
    """
    df = df.copy()
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'mom_{period}'] = df['close'].pct_change(period)
    
    # Volatility
    df['vol_10'] = df['close'].pct_change().rolling(10).std()
    df['vol_20'] = df['close'].pct_change().rolling(20).std()
    df['vol_ratio'] = df['vol_10'] / (df['vol_20'] + 1e-10)
    
    # EMAs
    for period in [20, 50]:
        df[f'ema{period}'] = df['close'].ewm(span=period).mean()
    
    # Trend features
    if 'ema20' in df.columns and 'ema50' in df.columns:
        df['trend'] = ((df['ema20'] > df['ema50']).astype(int) * 2 - 1)
        df['trend_str'] = abs(df['ema20'] - df['ema50']) / df['close']
        df['dist_ema50'] = (df['close'] - df['ema50']) / df['close']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = (df['rsi14'] - 50) / 50
    df['rsi_extreme'] = ((df['rsi14'] < 30) | (df['rsi14'] > 70)).astype(int)
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_pos'] = (df['bb_pct'] - 0.5) * 2
    
    # ADX
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (tr14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (tr14 + 1e-10))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()
    df['adx_strong'] = (df['adx'] > 25).astype(int)
    
    # Volume
    if 'volume' in df.columns:
        df['vol_surge'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    
    # Time-based features (for intraday models)
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['dow'] = df.index.dayofweek  # 0 = Monday, 6 = Sunday
    
    # Trading session (useful for FX)
    # Asian: 00:00-09:00 UTC, London: 08:00-17:00 UTC, NY: 13:00-22:00 UTC
    hour = df.index.hour
    df['session'] = 0  # Default: off-hours
    df.loc[(hour >= 0) & (hour < 9), 'session'] = 1  # Asian
    df.loc[(hour >= 8) & (hour < 17), 'session'] = 2  # London
    df.loc[(hour >= 13) & (hour < 22), 'session'] = 3  # NY
    
    # Position within session (0-1)
    df['session_pos'] = 0.0
    for session_id, start_hour, end_hour in [(1, 0, 9), (2, 8, 17), (3, 13, 22)]:
        mask = df['session'] == session_id
        session_hours = end_hour - start_hour
        df.loc[mask, 'session_pos'] = ((hour[mask] - start_hour) + (df.index.minute[mask] / 60)) / session_hours
    
    return df


def calculate_atr(df, period=14):
    """Calculate ATR."""
    if 'atr14' in df.columns:
        return df['atr14']
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def process_symbol(symbol: str):
    """
    Process one symbol using ensemble prediction.
    Combines models from multiple timeframes for a single symbol.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")
        
        # Load ensemble predictor
        ensemble = EnsemblePredictor(symbol)
        
        if ensemble.get_model_count() == 0:
            print(f"  ⚠️  No models available for {symbol}")
            return
        
        print(f"  ✅ Loaded {ensemble.get_model_count()} models: {ensemble.get_timeframes()}")
        
        # Fetch data (use 5T as reference timeframe for feature calculation)
        df = fetch_polygon_data(symbol, '5T', bars=200)
        if df is None or len(df) < 50:
            print(f"  ⚠️  Insufficient data")
            return
        
        # Calculate features
        df = calculate_features(df)
        df = df.dropna()
        
        if len(df) == 0:
            print(f"  ⚠️  No valid data after feature calculation")
            return
        
        # Get last row as features for prediction
        features_df = df.tail(1)
        
        # Get ensemble prediction
        print(f"  🤖 Generating ensemble prediction (strategy: {ENSEMBLE_STRATEGY})...")
        ensemble_result = ensemble.predict_ensemble(
            features_df, 
            strategy=ENSEMBLE_STRATEGY,
            min_models=2
        )
        
        if not ensemble_result:
            print(f"  ⚠️  Ensemble prediction failed")
            return
        
        signal_type = ensemble_result['prediction']
        confidence = ensemble_result['confidence']
        edge = ensemble_result['edge']
        
        print(f"  📊 Prediction: {signal_type.upper()}")
        print(f"  📊 Confidence: {confidence:.3f}")
        print(f"  📊 Edge: {edge:.3f}")
        print(f"  📊 Models: {ensemble_result['num_models']} total, {ensemble_result['agreeing_models']} agreeing")
        
        # Calculate TP/SL (use 5T parameters as default)
        atr = float(df['atr14'].iloc[-1]) if 'atr14' in df.columns else None
        if atr is None or pd.isna(atr) or atr == 0:
            atr = float(df['close'].iloc[-1]) * 0.02
        
        entry_price = float(df['close'].iloc[-1])
        params = SYMBOL_PARAMS.get(symbol, {}).get('5T', {'tp': 1.5, 'sl': 1.0})
        
        if signal_type == 'long':
            tp_price = entry_price + (atr * params['tp'])
            sl_price = entry_price - (atr * params['sl'])
        else:
            tp_price = entry_price - (atr * params['tp'])
            sl_price = entry_price + (atr * params['sl'])
        
        # Store signal in Supabase
        signal_response = supabase.table('live_signals').insert({
            'symbol': symbol,
            'timeframe': 'ensemble',  # Mark as ensemble signal
            'signal_type': signal_type,
            'confidence': float(confidence),
            'edge': float(edge),
            'entry_price': entry_price,
            'take_profit': tp_price,
            'stop_loss': sl_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        }).execute()
        
        # Get the signal ID
        signal_id = signal_response.data[0]['id'] if signal_response.data else None
        
        # Store ensemble metadata
        if signal_id:
            supabase.table('ensemble_metadata').insert({
                'signal_id': signal_id,
                'symbol': symbol,
                'timeframe': None,  # Ensemble across all timeframes
                'strategy': ENSEMBLE_STRATEGY,
                'num_models': ensemble_result['num_models'],
                'agreeing_models': ensemble_result['agreeing_models'],
                'votes': json.dumps(ensemble_result.get('votes', ensemble_result.get('weighted_votes', {}))),
                'model_predictions': json.dumps([{
                    'timeframe': p['timeframe'],
                    'prediction': p['prediction'],
                    'confidence': float(p['confidence']),
                    'edge': float(p['edge']),
                    'win_rate': float(p['win_rate'])
                } for p in ensemble_result['details']]),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }).execute()
        
        print(f"  ✅ {symbol}: {signal_type.upper()} @ {entry_price:.5f} (TP: {tp_price:.5f}, SL: {sl_price:.5f})")
        
    except Exception as e:
        print(f"  ❌ Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution."""
    print("\n" + "="*80)
    print(f"ENSEMBLE SIGNAL GENERATOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Strategy: {ENSEMBLE_STRATEGY}")
    print("="*80)
    
    success_count = 0
    for symbol in SYMBOLS:
        process_symbol(symbol)
        success_count += 1
        time.sleep(1)  # Rate limiting
    
    print("\n" + "="*80)
    print(f"✅ Processed {success_count}/{len(SYMBOLS)} symbols")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

