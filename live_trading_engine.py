#!/usr/bin/env python3
"""
LIVE TRADING ENGINE
===================
Fetches live data from Polygon, calculates features, generates signals.

IMPORTANT: Fixed look-ahead bias - entry price estimated as next bar open (current close + costs)
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

# Import unified cost model and guardrails
from market_costs import get_tp_sl, apply_entry_costs, calculate_tp_sl_prices
from execution_guardrails import get_moderate_guardrails

load_dotenv()

# Configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
API_URL = os.getenv('API_URL', 'http://localhost:8000')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Symbols and timeframes from your models
SYMBOLS = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
TIMEFRAMES = {
    '5T': {'minutes': 5, 'bars': 200},
    '15T': {'minutes': 15, 'bars': 200},
    '30T': {'minutes': 30, 'bars': 200},
    '1H': {'minutes': 60, 'bars': 200},
    '4H': {'minutes': 240, 'bars': 200},
}

# Polygon ticker mapping
TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD',
}

# TP/SL Parameters now unified in market_costs.py
# Old hardcoded SYMBOL_PARAMS removed to eliminate config drift


def fetch_polygon_data(symbol: str, timeframe_minutes: int, bars: int = 200):
    """Fetch OHLCV data from Polygon."""
    ticker = TICKER_MAP.get(symbol, symbol)
    
    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=timeframe_minutes * bars * 2)  # Extra buffer
    
    # Polygon API parameters
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': POLYGON_API_KEY
    }
    
    # Build URL for forex data
    multiplier = timeframe_minutes
    timespan = 'minute'
    
    from_date = start_time.strftime('%Y-%m-%d')
    to_date = end_time.strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    
    print(f"  Fetching {symbol} {timeframe_minutes}min from Polygon...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'OK' or not data.get('results'):
            print(f"  ⚠️  No data for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.set_index('timestamp').sort_index()
        
        print(f"  ✅ Got {len(df)} bars for {symbol}")
        return df
        
    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return None


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the same 30 features used in training."""
    df = df.copy()
    
    # Technical Indicators using pandas_ta
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    
    # Find actual column names (pandas_ta uses different names)
    col_mapping = {}
    for col in df.columns:
        if 'RSI' in col and '14' in col:
            col_mapping[col] = 'rsi14'
        elif 'EMA' in col and '20' in col:
            col_mapping[col] = 'ema20'
        elif 'EMA' in col and '50' in col:
            col_mapping[col] = 'ema50'
        elif 'EMA' in col and '200' in col:
            col_mapping[col] = 'ema200'
        elif 'MACD_' in col and not 's' in col.lower() and not 'h' in col.lower():
            col_mapping[col] = 'macd'
        elif 'MACDs' in col or 'MACD_S' in col:
            col_mapping[col] = 'macd_signal'
        elif 'MACDh' in col or 'MACD_H' in col:
            col_mapping[col] = 'macd_hist'
        elif 'BBL' in col:
            col_mapping[col] = 'bb_lower'
        elif 'BBM' in col:
            col_mapping[col] = 'bb_middle'
        elif 'BBU' in col:
            col_mapping[col] = 'bb_upper'
        elif 'ATR' in col and '14' in col:
            col_mapping[col] = 'atr14'
        elif 'ADX' in col and '14' in col and not 'DMP' in col and not 'DMN' in col:
            col_mapping[col] = 'adx'
    
    df = df.rename(columns=col_mapping)
    
    # Convert any DataFrame columns to Series (take first column)
    for col in ['rsi14', 'ema20', 'ema50', 'ema200', 'macd', 'macd_signal', 'macd_hist', 
                'bb_lower', 'bb_middle', 'bb_upper', 'atr14', 'adx']:
        if col in df.columns:
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                df[col] = col_data.iloc[:, 0]
    
    # Bollinger Band %
    if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
        try:
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        except:
            df['bb_pct'] = 0.5
    else:
        df['bb_pct'] = 0.5
    
    # Momentum
    for p in [5, 10, 20]:
        df[f'mom_{p}'] = df['close'].pct_change(p)
    
    # Volatility
    df['vol_10'] = df['close'].pct_change().rolling(10).std()
    df['vol_20'] = df['close'].pct_change().rolling(20).std()
    df['vol_ratio'] = df['vol_10'] / (df['vol_20'] + 1e-10)
    
    # Trend (with safe defaults)
    if 'ema20' in df.columns and 'ema50' in df.columns and 'atr14' in df.columns:
        try:
            df['trend'] = ((df['ema20'] > df['ema50']).astype(int) * 2 - 1)
            df['trend_str'] = abs(df['ema20'] - df['ema50']) / (df['atr14'] + 1e-10)
            df['dist_ema50'] = (df['close'] - df['ema50']) / (df['atr14'] + 1e-10)
        except:
            df['trend'] = 1
            df['trend_str'] = 0
            df['dist_ema50'] = 0
    else:
        df['trend'] = 1
        df['trend_str'] = 0
        df['dist_ema50'] = 0
    
    # RSI (with safe defaults)
    if 'rsi14' in df.columns:
        df['rsi_norm'] = (df['rsi14'] - 50) / 50
        df['rsi_extreme'] = ((df['rsi14'] < 30) | (df['rsi14'] > 70)).astype(int)
    else:
        df['rsi14'] = 50
        df['rsi_norm'] = 0
        df['rsi_extreme'] = 0
    
    # ADX (with safe defaults)
    if 'adx' in df.columns:
        try:
            df['adx_strong'] = (df['adx'] > 25).astype(int)
        except:
            df['adx'] = 20
            df['adx_strong'] = 0
    else:
        df['adx'] = 20
        df['adx_strong'] = 0
    
    # Volume (if available)
    if 'volume' in df.columns:
        df['vol_sma'] = df['volume'].rolling(20).mean()
        df['vol_ratio_price'] = df['volume'] / (df['vol_sma'] + 1)
    else:
        df['vol_sma'] = 0
        df['vol_ratio_price'] = 1
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


def extract_feature_vector(df: pd.DataFrame) -> list:
    """Extract the 30 features for the latest bar."""
    # Feature list (same order as training)
    features = [
        'mom_5', 'mom_10', 'mom_20',
        'vol_10', 'vol_20', 'vol_ratio',
        'trend', 'trend_str', 'dist_ema50',
        'rsi14', 'rsi_norm', 'rsi_extreme',
        'adx', 'adx_strong',
        'macd', 'macd_signal', 'macd_hist',
        'bb_pct', 'bb_upper', 'bb_lower',
        'ema20', 'ema50', 'ema200',
        'atr14',
        'vol_sma', 'vol_ratio_price',
        'close', 'high', 'low', 'open'
    ]
    
    # Ensure MACD columns exist with defaults
    for col in ['macd', 'macd_signal', 'macd_hist']:
        if col not in df.columns:
            df[col] = 0.0
    
    # Ensure EMA columns exist with defaults
    for col in ['ema20', 'ema50', 'ema200']:
        if col not in df.columns:
            df[col] = df['close']
    
    # Ensure BB columns exist with defaults
    for col in ['bb_upper', 'bb_lower']:
        if col not in df.columns:
            df[col] = df['close']
    
    # Ensure ATR exists
    if 'atr14' not in df.columns:
        df['atr14'] = df['close'] * 0.02
    
    # Get latest row
    latest = df.iloc[-1]
    
    # Extract values
    feature_vector = []
    for feat in features:
        if feat in latest.index:
            val = latest[feat]
            # Convert Series/DataFrame to scalar
            if isinstance(val, (pd.Series, pd.DataFrame)):
                val = val.iloc[0] if len(val) > 0 else 0.0
            # Handle inf/nan
            try:
                if pd.isna(val) or np.isinf(val):
                    val = 0.0
            except:
                val = 0.0
            feature_vector.append(float(val))
        else:
            feature_vector.append(0.0)
    
    return feature_vector


def get_prediction(symbol: str, timeframe: str, features: list) -> dict:
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                'symbol': symbol,
                'timeframe': timeframe,
                'features': features
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ❌ Prediction error: {e}")
        return None


# calculate_tp_sl_prices() now imported from market_costs.py (unified cost model)


def store_signal_in_supabase(signal_data: dict):
    """Store signal in Supabase."""
    try:
        # Store in live_signals table
        supabase.table('live_signals').insert({
            'symbol': signal_data['symbol'],
            'timeframe': signal_data['timeframe'],
            'signal_type': signal_data['directional_signal'],
            'confidence': signal_data['confidence'],
            'edge': signal_data['edge'],
            'entry_price': signal_data['current_price'],
            'take_profit': signal_data.get('tp_price'),
            'stop_loss': signal_data.get('sl_price'),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        }).execute()
        
        print(f"  💾 Stored {signal_data['symbol']} {signal_data['timeframe']} signal")
        print(f"     Entry: {signal_data['current_price']:.5f} | TP: {signal_data.get('tp_price', 0):.5f} | SL: {signal_data.get('sl_price', 0):.5f}")
    except Exception as e:
        print(f"  ❌ Storage error: {e}")


def process_symbol_timeframe(symbol: str, timeframe: str):
    """Process one symbol/timeframe combination."""
    print(f"\n{'='*60}")
    print(f"Processing {symbol} {timeframe}")
    print(f"{'='*60}")
    
    # Fetch data
    tf_config = TIMEFRAMES[timeframe]
    df = fetch_polygon_data(symbol, tf_config['minutes'], tf_config['bars'])
    
    if df is None or len(df) < 100:
        print(f"  ⚠️  Insufficient data")
        return
    
    # Calculate features
    print(f"  📊 Calculating features...")
    df = calculate_features(df)
    
    if len(df) < 50:
        print(f"  ⚠️  Insufficient data after feature calculation")
        return
    
    # Extract feature vector
    features = extract_feature_vector(df)

    # IMPORTANT: In live trading, we get signal on current bar close,
    # but actual entry will be at NEXT bar open (unknown at signal time).
    # We use current close as proxy and apply realistic entry costs.
    current_close = float(df.iloc[-1]['close'])
    last_bar_time = df.index[-1]

    print(f"  📈 Current price: {current_close:.5f}")
    print(f"  🔢 Features: {len(features)}")

    # Get prediction
    print(f"  🤖 Getting prediction...")
    prediction = get_prediction(symbol, timeframe, features)

    if prediction is None:
        return

    # Display result
    signal = prediction['directional_signal']
    quality = prediction['signal_quality']
    confidence = prediction['confidence']

    quality_icon = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}[quality]

    print(f"  {quality_icon} Signal: {signal.upper()}")
    print(f"  📊 Quality: {quality.upper()}")
    print(f"  💪 Confidence: {confidence*100:.1f}%")
    print(f"  📈 Should Trade: {prediction['should_trade']}")

    # Calculate ATR
    atr = float(df.iloc[-1].get('atr14', current_close * 0.02))  # Use 2% of price as fallback

    # Apply execution guardrails (staleness, spread, volatility, session, confidence)
    print(f"  🛡️  Checking execution guardrails...")
    guardrails = get_moderate_guardrails()

    # Get timeframe in minutes
    tf_minutes = TIMEFRAMES[timeframe]['minutes']

    # Estimate spread (in practice, get from broker's live quote)
    from market_costs import get_costs
    costs = get_costs(symbol)

    # Convert spread pips to price
    if symbol == 'XAUUSD':
        current_spread = costs.spread_pips * 0.10  # 1 pip = $0.10
    elif symbol == 'XAGUSD':
        current_spread = costs.spread_pips * 0.01  # 1 pip = $0.01
    else:  # Forex
        current_spread = costs.spread_pips * 0.0001  # 1 pip = 0.0001

    guardrail_results = guardrails.check_all(
        last_bar_time=last_bar_time,
        timeframe_minutes=tf_minutes,
        current_spread=current_spread,
        atr=atr,
        price=current_close,
        confidence=confidence
    )

    # Check if guardrails passed
    if not guardrails.all_passed(guardrail_results):
        failures = guardrails.get_failures(guardrail_results)
        print(f"  ❌ Guardrails FAILED - Signal blocked:")
        for name, reason in failures.items():
            print(f"     • {name}: {reason}")
        return

    print(f"  ✅ All guardrails passed")

    # Estimate entry price with costs (next bar open ≈ current close + costs)
    # In live trading, actual entry will be at next bar open
    notional = 100000  # $100k position for cost calculation
    estimated_entry, entry_commission, entry_slippage = apply_entry_costs(
        symbol, current_close, notional, direction=signal
    )

    print(f"  💰 Estimated entry: {estimated_entry:.5f} (includes spread + commission + slippage)")

    # Calculate TP/SL prices using unified cost model
    tp_price, sl_price = calculate_tp_sl_prices(symbol, timeframe, estimated_entry, signal, atr)

    # Store signal
    signal_data = {
        **prediction,
        'current_price': current_close,
        'estimated_entry': estimated_entry,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'atr': atr,
        'spread': current_spread,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

    store_signal_in_supabase(signal_data)

    # Alert on high-quality signals
    if prediction['should_trade']:
        print(f"\n  🚨 HIGH QUALITY SIGNAL! {signal.upper()} {symbol} {timeframe}")
        print(f"     Entry: {estimated_entry:.5f} | TP: {tp_price:.5f} | SL: {sl_price:.5f}")


def run_once():
    """Run one complete cycle for all models."""
    print("\n" + "="*80)
    print(f"LIVE TRADING ENGINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Get available models from API
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        available_models = response.json()
        print(f"✅ {len(available_models)} models available")
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        return
    
    # Process each model
    for model_key in available_models:
        symbol, timeframe = model_key.split('_')
        
        try:
            process_symbol_timeframe(symbol, timeframe)
        except Exception as e:
            import traceback
            print(f"❌ Error processing {symbol} {timeframe}: {e}")
            if "--debug" in sys.argv:
                traceback.print_exc()
        
        # Rate limiting
        time.sleep(2)


def run_continuous(interval_minutes: int = 5):
    """Run continuously with specified interval."""
    print("\n" + "="*80)
    print("LIVE TRADING ENGINE - CONTINUOUS MODE")
    print("="*80)
    print(f"Interval: {interval_minutes} minutes")
    print(f"Press Ctrl+C to stop")
    print("="*80)
    
    while True:
        try:
            run_once()
            
            print(f"\n⏰ Sleeping for {interval_minutes} minutes...")
            print(f"Next run at: {(datetime.now() + timedelta(minutes=interval_minutes)).strftime('%H:%M:%S')}")
            
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping live trading engine...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"Retrying in 1 minute...")
            time.sleep(60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        run_once()
    else:
        run_continuous(interval_minutes=5)

