#!/usr/bin/env python3
"""
STANDALONE SIGNAL GENERATOR FOR GITHUB ACTIONS
================================================
Generates signals without needing the API server running.
Uses ensemble predictions from multiple models.

IMPORTANT: Fixed to use unified cost model and execution guardrails
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client
import pandas_ta as ta

# Import ensemble predictor and news filter
sys.path.insert(0, str(Path(__file__).parent))

# Import BalancedModel FIRST before loading any pickled models
from balanced_model import BalancedModel
from production_final_system import BalancedModel as BalancedModelAlt  # Compatibility

from ensemble_predictor import EnsemblePredictor
from news_filter import is_in_blackout_window
from live_feature_utils import build_feature_frame

# Import unified cost model and guardrails
from market_costs import get_tp_sl, apply_entry_costs, calculate_tp_sl_prices, get_costs
from execution_guardrails import get_moderate_guardrails

# Ensure BalancedModel is available for pickle in all contexts
_main_module = sys.modules.get("__main__")
if _main_module is not None and not hasattr(_main_module, "BalancedModel"):
    setattr(_main_module, "BalancedModel", BalancedModel)

# Also make it available under production_final_system module name for compatibility
import production_final_system
if not hasattr(production_final_system, "BalancedModel"):
    setattr(production_final_system, "BalancedModel", BalancedModel)

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

if not all([POLYGON_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    print("❌ Missing required environment variables!")
    print(f"   POLYGON_API_KEY: {'✅' if POLYGON_API_KEY else '❌'}")
    print(f"   SUPABASE_URL: {'✅' if SUPABASE_URL else '❌'}")
    print(f"   SUPABASE_KEY: {'✅' if SUPABASE_KEY else '❌'}")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Models to process (from your production system)
# NOTE: 4H timeframes temporarily disabled - Polygon REST API doesn't provide real-time 4H forex data
# FOCUS: Only XAUUSD and XAGUSD for now - other symbols disabled until further testing
MODELS = [
    # XAGUSD models (5 timeframes including 4H)
    ('XAGUSD', '5T'), ('XAGUSD', '15T'), ('XAGUSD', '30T'), ('XAGUSD', '1H'), ('XAGUSD', '4H'),
    # XAUUSD models (4 timeframes)
    ('XAUUSD', '5T'), ('XAUUSD', '15T'), ('XAUUSD', '30T'), ('XAUUSD', '1H'),
]

# Disabled models - will re-enable after XAUUSD/XAGUSD are stable
MODELS_DISABLED = [
    ('AUDUSD', '15T'), ('AUDUSD', '30T'), ('AUDUSD', '5T'), ('AUDUSD', '1H'),
    ('EURUSD', '30T'), ('EURUSD', '5T'),
    ('GBPUSD', '15T'), ('GBPUSD', '1H'), ('GBPUSD', '30T'), ('GBPUSD', '5T'),
    ('NZDUSD', '15T'), ('NZDUSD', '1H'), ('NZDUSD', '30T'), ('NZDUSD', '5T'),
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

# TP/SL Parameters now unified in market_costs.py
# Old hardcoded SYMBOL_PARAMS removed to eliminate config drift

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
    """Fetch OHLCV data from Polygon REST API (PAID PLAN - REAL-TIME)."""
    ticker = TICKER_MAP.get(symbol, symbol)
    minutes = TIMEFRAME_MINUTES[timeframe]
    
    # For 4H, fetch 1H bars and resample (4H bars are stale on API)
    if timeframe == '4H':
        fetch_minutes = 60  # Fetch 1H bars
        bars_to_fetch = bars * 8  # Need 8x more to ensure fresh data after resampling
    else:
        fetch_minutes = minutes
        bars_to_fetch = bars
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=fetch_minutes * bars_to_fetch * 2)
    
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }
    
    multiplier = fetch_minutes
    timespan = 'minute'
    from_date = start_time.strftime('%Y-%m-%d')
    to_date = end_time.strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'results' not in data or not data['results']:
            return None
            
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')
        df = df.sort_index()
        
        # Resample to 4H if needed
        if timeframe == '4H':
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        return df.tail(bars)
        
    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return None


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
        
        # For 4H, allow up to 24 hours staleness (S3 files update overnight)
        if timeframe == '4H':
            max_allowed = timedelta(hours=24)
        else:
            max_allowed = timedelta(minutes=TIMEFRAME_MINUTES[timeframe] * 2)  # Default: 2x timeframe
            # For low-volume periods, allow up to 6 hours
            max_allowed = max(max_allowed, timedelta(hours=6))
            
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

        # IMPORTANT: Entry will be at next bar open (unknown at signal time)
        # Use current close as proxy and apply realistic entry costs
        current_close = float(raw_df['close'].iloc[-1])
        last_bar_time = raw_df.index[-1]

        # Apply execution guardrails
        print(f"  🛡️  Checking execution guardrails...")
        guardrails = get_moderate_guardrails()
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
            timeframe_minutes=TIMEFRAME_MINUTES[timeframe],
            current_spread=current_spread,
            atr=atr,
            price=current_close,
            confidence=confidence
        )

        # Check if guardrails passed
        if not guardrails.all_passed(guardrail_results):
            failures = guardrails.get_failures(guardrail_results)
            print(f"  ❌ {symbol} {timeframe}: Guardrails FAILED - Signal blocked:")
            for name, reason in failures.items():
                print(f"     • {name}: {reason}")
            return

        print(f"  ✅ {symbol} {timeframe}: All guardrails passed")

        # Estimate entry price with costs (next bar open ≈ current close + costs)
        notional = 100000  # $100k position for cost calculation
        estimated_entry, entry_commission, entry_slippage = apply_entry_costs(
            symbol, current_close, notional, direction=signal_type
        )

        # Calculate TP/SL using unified cost model
        tp_price, sl_price = calculate_tp_sl_prices(symbol, timeframe, estimated_entry, signal_type, atr)

        supabase_payload = {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': signal_type,
            'confidence': float(confidence),
            'edge': float(edge),
            'entry_price': estimated_entry,  # Now includes costs
            'take_profit': tp_price,
            'stop_loss': sl_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        supabase.table('live_signals').insert(supabase_payload).execute()

        print(f"  ✅ {symbol} {timeframe}: {signal_type.upper()} @ {estimated_entry:.5f} (TP: {tp_price:.5f}, SL: {sl_price:.5f})")
    except Exception as e:
        print(f"  ❌ Error processing {symbol} {timeframe}: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to be caught by main() error handler


def main():
    """Main execution - generates signals for all production models."""
    print("\n" + "="*80)
    print(f"STANDALONE SIGNAL GENERATOR - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*80 + "\n")

    success_count = 0
    error_count = 0
    errors = []

    for symbol, timeframe in MODELS:
        try:
            process_symbol(symbol, timeframe)
            success_count += 1
        except Exception as e:
            error_count += 1
            error_msg = f"{symbol} {timeframe}: {str(e)}"
            errors.append(error_msg)
            print(f"  ❌ ERROR: {error_msg}")
        time.sleep(0.5)

    print("\n" + "="*80)
    print(f"SIGNAL GENERATION COMPLETE")
    print(f"  ✅ Success: {success_count}/{len(MODELS)}")
    if error_count > 0:
        print(f"  ❌ Errors: {error_count}")
        print(f"\nError details:")
        for err in errors:
            print(f"  • {err}")
    print("="*80 + "\n")

    # Calculate success rate
    success_rate = (success_count / len(MODELS) * 100) if len(MODELS) > 0 else 0

    # Only fail if success rate is critically low (< 80%)
    # This prevents workflow failures from individual symbol issues
    if error_count == len(MODELS):
        print("❌ CRITICAL: All models failed!")
        sys.exit(1)
    elif success_rate < 80:
        print(f"❌ CRITICAL: Success rate too low ({success_rate:.1f}% < 80%)")
        sys.exit(1)
    elif error_count > 0:
        print(f"⚠️  Partial failure: {error_count}/{len(MODELS)} models failed ({success_rate:.1f}% success)")
        print("   This is acceptable - some symbols may have stale data or be in blackout")
        # Exit with success since majority of models worked

    print(f"\n✅ Signal generation completed successfully ({success_rate:.1f}% success rate)")
    sys.exit(0)


if __name__ == "__main__":
    main()

