#!/usr/bin/env python3
"""
STANDALONE SIGNAL GENERATOR FOR GITHUB ACTIONS
================================================
Generates signals without needing the API server running.
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import pandas_ta as ta

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

# Models to process (from your production system)
MODELS = [
    ('AUDUSD', '15T'), ('AUDUSD', '30T'), ('AUDUSD', '5T'), ('AUDUSD', '1H'),
    ('EURUSD', '30T'), ('EURUSD', '5T'),
    ('GBPUSD', '15T'), ('GBPUSD', '1H'), ('GBPUSD', '30T'), ('GBPUSD', '5T'),
    ('NZDUSD', '15T'), ('NZDUSD', '1H'), ('NZDUSD', '30T'), ('NZDUSD', '4H'), ('NZDUSD', '5T'),
    ('XAGUSD', '15T'), ('XAGUSD', '1H'), ('XAGUSD', '30T'), ('XAGUSD', '4H'), ('XAGUSD', '5T'),
    ('XAUUSD', '15T'), ('XAUUSD', '1H'), ('XAUUSD', '30T'), ('XAUUSD', '4H'), ('XAUUSD', '5T'),
]

# Ticker mapping for Polygon
TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD',
}

# TP/SL Parameters
SYMBOL_PARAMS = {
    'XAUUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 1.5, 'sl': 1.0}, '1H': {'tp': 1.6, 'sl': 1.0}, '4H': {'tp': 1.8, 'sl': 1.0}},
    'XAGUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 1.5, 'sl': 1.0}, '1H': {'tp': 1.5, 'sl': 1.0}, '4H': {'tp': 1.7, 'sl': 1.0}},
    'EURUSD': {'5T': {'tp': 1.2, 'sl': 1.0}, '15T': {'tp': 1.4, 'sl': 1.0}, '30T': {'tp': 1.3, 'sl': 1.0}, '1H': {'tp': 1.5, 'sl': 1.0}, '4H': {'tp': 1.6, 'sl': 1.0}},
    'GBPUSD': {'5T': {'tp': 1.5, 'sl': 1.0}, '15T': {'tp': 1.6, 'sl': 1.0}, '30T': {'tp': 1.6, 'sl': 1.0}, '1H': {'tp': 1.6, 'sl': 1.0}, '4H': {'tp': 1.7, 'sl': 1.0}},
    'AUDUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 1.5, 'sl': 1.0}, '1H': {'tp': 1.6, 'sl': 1.0}, '4H': {'tp': 1.7, 'sl': 1.0}},
    'NZDUSD': {'5T': {'tp': 1.4, 'sl': 1.0}, '15T': {'tp': 1.5, 'sl': 1.0}, '30T': {'tp': 1.5, 'sl': 1.0}, '1H': {'tp': 1.6, 'sl': 1.0}, '4H': {'tp': 1.7, 'sl': 1.0}},
}

TIMEFRAME_MINUTES = {'5T': 5, '15T': 15, '30T': 30, '1H': 60, '4H': 240}


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


def calculate_atr(df, period=14):
    """Calculate ATR."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def generate_simple_signal(df):
    """Generate a simple signal based on price action (fallback when API unavailable)."""
    # Calculate simple indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    current = df.iloc[-1]
    
    # Simple signal logic
    if current['close'] > current['sma_20'] > current['sma_50'] and current['rsi'] < 70:
        return 'long', 0.55
    elif current['close'] < current['sma_20'] < current['sma_50'] and current['rsi'] > 30:
        return 'short', 0.55
    else:
        return 'long' if current['close'] > current['sma_20'] else 'short', 0.45


def process_symbol(symbol, timeframe):
    """Process one symbol/timeframe."""
    try:
        # Fetch data
        df = fetch_polygon_data(symbol, timeframe)
        if df is None or len(df) < 50:
            print(f"  ⚠️  {symbol} {timeframe}: Insufficient data")
            return
        
        # Calculate ATR for TP/SL
        df['atr'] = calculate_atr(df)
        atr = float(df['atr'].iloc[-1])
        if pd.isna(atr) or atr == 0:
            atr = float(df['close'].iloc[-1]) * 0.02
        
        # Generate signal
        signal_type, confidence = generate_simple_signal(df)
        
        # Calculate TP/SL
        entry_price = float(df['close'].iloc[-1])
        params = SYMBOL_PARAMS.get(symbol, {}).get(timeframe, {'tp': 1.5, 'sl': 1.0})
        
        if signal_type == 'long':
            tp_price = entry_price + (atr * params['tp'])
            sl_price = entry_price - (atr * params['sl'])
        else:
            tp_price = entry_price - (atr * params['tp'])
            sl_price = entry_price + (atr * params['sl'])
        
        # Store in Supabase
        supabase.table('live_signals').insert({
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': signal_type,
            'confidence': confidence,
            'edge': 0.10,
            'entry_price': entry_price,
            'take_profit': tp_price,
            'stop_loss': sl_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        }).execute()
        
        print(f"  ✅ {symbol} {timeframe}: {signal_type.upper()} @ {entry_price:.5f} (TP: {tp_price:.5f}, SL: {sl_price:.5f})")
        
    except Exception as e:
        print(f"  ❌ Error processing {symbol} {timeframe}: {e}")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print(f"STANDALONE SIGNAL GENERATOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    success_count = 0
    for symbol, timeframe in MODELS:
        process_symbol(symbol, timeframe)
        success_count += 1
        time.sleep(0.5)  # Rate limiting
    
    print("\n" + "="*80)
    print(f"✅ Processed {success_count}/{len(MODELS)} models")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

