#!/usr/bin/env python3
"""
STANDALONE SIGNAL GENERATOR FOR GITHUB ACTIONS (FIXED)
=======================================================
Fixes:
1. 4H staleness: Validate 1H freshness before resampling
2. Missing volume features: Defensive volume engineering with fallbacks
3. 1H insufficient bars: Fetch more bars, drop fewer in feature engineering
4. API latency: Account for 1-2 bar lag in staleness checks

CHANGELOG:
- Added pre-resampling freshness check for 4H
- Volume features now use fallback values when data is unreliable
- Increased fetch bars and relaxed min requirements for 1H/4H
- Staleness tolerance now accounts for bar close + API lag
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
from production_final_system import BalancedModel as BalancedModelAlt

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

import production_final_system
if not hasattr(production_final_system, "BalancedModel"):
    setattr(production_final_system, "BalancedModel", BalancedModel)

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

# Models to process
MODELS = [
    # XAGUSD models (5 timeframes including 4H)
    ('XAGUSD', '5T'), ('XAGUSD', '15T'), ('XAGUSD', '30T'), ('XAGUSD', '1H'), ('XAGUSD', '4H'),
    # XAUUSD models (4 timeframes)
    ('XAUUSD', '5T'), ('XAUUSD', '15T'), ('XAUUSD', '30T'), ('XAUUSD', '1H'),
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

TIMEFRAME_MINUTES = {'5T': 5, '15T': 15, '30T': 30, '1H': 60, '4H': 240}

# INCREASED bars to account for feature engineering dropping rows
BARS_PER_TF = {
    '5T': 500,   # Was 400, need more for long SMAs
    '15T': 350,  # Was 240
    '30T': 250,  # Was 160
    '1H': 250,   # Was 120 - CRITICAL: Need 200+ for SMA200
    '4H': 150,   # Was 80 - Need more for 4H features after resampling
}

# RELAXED minimum requirements (feature engineering will still drop some)
MIN_BARS_REQUIRED = {
    '5T': 100,   # Was 120
    '15T': 100,  # Was 120
    '30T': 80,   # Was 120
    '1H': 60,    # Was 80 - Relaxed since we fetch more now
    '4H': 30,    # Was 40 - Relaxed for 4H
}

# Cache for ensemble predictors
ENSEMBLE_CACHE = {}

# Sentiment filtering thresholds
SENTIMENT_LONG_THRESHOLD = 0.2
SENTIMENT_SHORT_THRESHOLD = -0.2


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
            return 0.0
    
    except Exception as e:
        print(f"  ⚠️  Error fetching sentiment for {symbol}: {e}")
        return 0.0


def fetch_polygon_data(symbol: str, timeframe: str, bars: int = 200):
    """
    Fetch OHLCV data from Polygon REST API.
    
    FIXES:
    - For 4H: Validate 1H freshness BEFORE resampling
    - Account for API lag in staleness calculations
    """
    ticker = TICKER_MAP.get(symbol, symbol)
    minutes = TIMEFRAME_MINUTES[timeframe]
    
    # For 4H, fetch 1H bars and resample
    if timeframe == '4H':
        fetch_minutes = 60
        bars_to_fetch = bars * 8  # 8x more for buffer after resampling
    else:
        fetch_minutes = minutes
        bars_to_fetch = bars
    
    end_time = datetime.now(timezone.utc)
    # Fetch extra history to ensure we have enough after feature engineering
    start_time = end_time - timedelta(minutes=fetch_minutes * bars_to_fetch * 3)

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    multiplier = fetch_minutes
    timespan = 'minute'

    # Use timestamps for real-time data
    from_timestamp = int(start_time.timestamp() * 1000)
    to_timestamp = int(end_time.timestamp() * 1000)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_timestamp}/{to_timestamp}"
    
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
        
        # FIX #1: For 4H, validate 1H freshness BEFORE resampling
        if timeframe == '4H':
            if len(df) == 0:
                print(f"  ❌ {symbol} 4H: No 1H data received from API")
                return None
                
            last_1h_bar = df.index[-1]
            now = datetime.now(timezone.utc)
            lag_1h = now - last_1h_bar
            
            # 1H bars should be fresh within 2 hours (1 bar + API lag)
            if lag_1h > timedelta(hours=2):
                print(f"  ❌ {symbol} 4H: 1H source data is stale (last bar {last_1h_bar}, Δ {lag_1h})")
                print(f"     Cannot resample stale 1H → 4H. Skipping.")
                return None
            
            print(f"  ✅ {symbol} 4H: 1H source fresh (last bar {last_1h_bar}, Δ {lag_1h})")
            
            # Now resample to 4H
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        return df.tail(bars)
        
    except Exception as e:
        print(f"  ❌ Error fetching {symbol} {timeframe}: {e}")
        return None


def add_volume_features_defensive(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX #2: Add volume features with defensive handling for unreliable volume.
    
    For XAUUSD/XAGUSD, volume from Polygon spot is often 0 or unreliable.
    Use synthetic fallbacks to prevent NaN propagation.
    """
    df = df.copy()
    
    # Check if volume is all zeros or all NaN
    volume_usable = (df['volume'] > 0).sum() > len(df) * 0.5  # >50% non-zero
    
    if not volume_usable:
        print(f"  ⚠️  Volume unreliable (mostly zero/NaN), using synthetic fallback")
        # Create synthetic volume based on price movement
        df['volume'] = (df['high'] - df['low']).rolling(20).mean().fillna(1.0)
    
    # Now calculate volume features with zero-division protection
    volume_sma20 = df['volume'].rolling(20).mean()
    volume_sma20 = volume_sma20.replace(0, np.nan).fillna(method='bfill').fillna(1.0)
    
    df['volume_sma20'] = volume_sma20
    df['volume_ratio'] = df['volume'] / volume_sma20.replace(0, 1.0)  # Avoid div by zero
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    return df


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
        min_bars = MIN_BARS_REQUIRED.get(timeframe, 100)
        if raw_df is None or len(raw_df) < min_bars:
            got = 0 if raw_df is None else len(raw_df)
            print(f"  ⚠️  {symbol} {timeframe}: Insufficient data (have {got} < {min_bars} bars)")
            return

        # FIX #3: Add defensive volume features BEFORE build_feature_frame
        raw_df = add_volume_features_defensive(raw_df)

        last_bar_time = raw_df.index[-1]
        staleness = datetime.now(timezone.utc) - last_bar_time
        
        # FIX #4: More realistic staleness tolerances accounting for bar close + API lag
        if timeframe == '4H':
            # 4H bars update every 4 hours. Allow 5H for bar close + API lag
            max_allowed = timedelta(hours=5)
        elif timeframe == '1H':
            # 1H bars: Allow 90 minutes (bar close + lag)
            max_allowed = timedelta(minutes=90)
        else:
            # Shorter timeframes: 3x the bar period (1 current + 1 lag + 1 buffer)
            max_allowed = timedelta(minutes=TIMEFRAME_MINUTES[timeframe] * 3)
            
        if staleness > max_allowed:
            print(f"  ⚠️  {symbol} {timeframe}: Stale data (last bar {last_bar_time} UTC, Δ {staleness})")
            return

        # Now build features - volume features already added above
        feature_df = build_feature_frame(raw_df)
        if feature_df.empty:
            print(f"  ⚠️  {symbol} {timeframe}: Unable to build feature set (insufficient history or NaNs)")
            return

        # Diagnostic: Check final feature count
        print(f"  📊 {symbol} {timeframe}: Feature engineering: {len(raw_df)} → {len(feature_df)} bars")

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

        current_close = float(raw_df['close'].iloc[-1])
        last_bar_time = raw_df.index[-1]

        # Apply execution guardrails
        print(f"  🛡️  Checking execution guardrails...")
        guardrails = get_moderate_guardrails()
        costs = get_costs(symbol)

        # Convert spread pips to price
        if symbol == 'XAUUSD':
            current_spread = costs.spread_pips * 0.10
        elif symbol == 'XAGUSD':
            current_spread = costs.spread_pips * 0.01
        else:
            current_spread = costs.spread_pips * 0.0001

        guardrail_results = guardrails.check_all(
            last_bar_time=last_bar_time,
            timeframe_minutes=TIMEFRAME_MINUTES[timeframe],
            current_spread=current_spread,
            atr=atr,
            price=current_close,
            confidence=confidence
        )

        if not guardrails.all_passed(guardrail_results):
            failures = guardrails.get_failures(guardrail_results)
            print(f"  ❌ {symbol} {timeframe}: Guardrails FAILED - Signal blocked:")
            for name, reason in failures.items():
                print(f"     • {name}: {reason}")
            return

        print(f"  ✅ {symbol} {timeframe}: All guardrails passed")

        # Estimate entry price with costs
        notional = 100000
        estimated_entry, entry_commission, entry_slippage = apply_entry_costs(
            symbol, current_close, notional, direction=signal_type
        )

        # Calculate TP/SL using unified cost model
        tp_price, sl_price = calculate_tp_sl_prices(symbol, timeframe, estimated_entry, atr, signal_type)

        supabase_payload = {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal_type': signal_type,
            'confidence': float(confidence),
            'edge': float(edge),
            'entry_price': estimated_entry,
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
        raise


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

    success_rate = (success_count / len(MODELS) * 100) if len(MODELS) > 0 else 0

    if error_count == len(MODELS):
        print("❌ CRITICAL: All models failed!")
        sys.exit(1)
    elif success_rate < 80:
        print(f"❌ CRITICAL: Success rate too low ({success_rate:.1f}% < 80%)")
        sys.exit(1)
    elif error_count > 0:
        print(f"⚠️  Partial failure: {error_count}/{len(MODELS)} models failed ({success_rate:.1f}% success)")
        print("   This is acceptable - some symbols may have stale data or be in blackout")

    print(f"\n✅ Signal generation completed successfully ({success_rate:.1f}% success rate)")
    sys.exit(0)


if __name__ == "__main__":
    main()
