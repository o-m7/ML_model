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
# FOCUS: Only XAUUSD and XAGUSD for now - other symbols disabled until further testing
MODELS = [
    # XAGUSD models (4 timeframes - no 4H model exists)
    ('XAGUSD', '5T'), ('XAGUSD', '15T'), ('XAGUSD', '30T'), ('XAGUSD', '1H'),
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

TIMEFRAME_MINUTES = {'5T': 5, '15T': 15, '30T': 30, '1H': 60}

# ALWAYS fetch maximum bars from Polygon (50,000 limit)
# This provides rich historical context for feature calculations
MAX_BARS_FROM_API = 50000

# Minimum bars needed per timeframe (safety check)
MIN_BARS_REQUIRED = {
    '5T': 120,
    '15T': 120,
    '30T': 120,
    '1H': 80,
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


def fetch_live_price(symbol: str) -> Optional[float]:
    """
    Fetch CURRENT LIVE price (latest tick) from Polygon.
    This is separate from historical bars - ensures we trade on current price.
    """
    ticker = TICKER_MAP.get(symbol, symbol)

    try:
        # Get the absolute latest 1-minute bar (most current price available)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=5)  # Last 5 minutes to ensure we get data

        from_timestamp = int(start_time.timestamp() * 1000)
        to_timestamp = int(end_time.timestamp() * 1000)

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{from_timestamp}/{to_timestamp}"

        params = {
            'adjusted': 'true',
            'sort': 'desc',  # Descending to get latest first
            'limit': 1,  # Only need the latest bar
            'apiKey': POLYGON_API_KEY
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'results' in data and data['results'] and len(data['results']) > 0:
            latest_bar = data['results'][0]
            live_price = float(latest_bar['c'])  # Close of latest bar
            bar_time = pd.to_datetime(latest_bar['t'], unit='ms', utc=True)

            # Verify it's actually live (within last 5 minutes)
            age_seconds = (datetime.now(timezone.utc) - bar_time).total_seconds()
            if age_seconds > 300:  # More than 5 minutes old
                print(f"  ⚠️  Live price is {age_seconds:.0f}s old, may not be current")

            return live_price
        else:
            print(f"  ⚠️  No live price data available for {symbol}")
            return None

    except Exception as e:
        print(f"  ❌ Error fetching live price for {symbol}: {e}")
        return None


def fetch_polygon_data(symbol: str, timeframe: str):
    """Fetch MAXIMUM OHLCV data from Polygon REST API (50,000 bars) for FEATURE CALCULATION."""
    ticker = TICKER_MAP.get(symbol, symbol)
    minutes = TIMEFRAME_MINUTES[timeframe]

    # Calculate lookback to get MAX_BARS_FROM_API
    fetch_minutes = minutes
    lookback_days = (MAX_BARS_FROM_API * fetch_minutes) // (60 * 24)

    end_time = datetime.now(timezone.utc)
    # Go back far enough to get 50k bars (with buffer for weekends/holidays)
    start_time = end_time - timedelta(days=lookback_days + 30)

    print(f"  🔍 [{end_time.strftime('%H:%M:%S')}] Fetching {timeframe} data (via {fetch_minutes}min bars, ~{lookback_days} days)...")

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    multiplier = fetch_minutes
    timespan = 'minute'

    # CRITICAL FIX: Use timestamps instead of dates to get LIVE data
    # Dates only give you data up to a cutoff time, not current minute
    from_timestamp = int(start_time.timestamp() * 1000)  # milliseconds
    to_timestamp = int(end_time.timestamp() * 1000)      # milliseconds

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

        # Debug: Log data freshness and volume
        if not df.empty:
            last_bar = df.index[-1]
            age_seconds = (end_time - last_bar).total_seconds()
            age_hours = age_seconds / 3600
            total_bars = len(df)
            first_bar = df.index[0]
            print(f"  📊 Fetched {total_bars:,} bars: {first_bar.strftime('%Y-%m-%d')} to {last_bar.strftime('%Y-%m-%d %H:%M UTC')} (age: {age_hours:.1f}h)")

        return df  # Return ALL data for maximum feature calculation context
        
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

        # Fetch ALL available data (up to 50k bars) for rich feature context
        raw_df = fetch_polygon_data(symbol, timeframe)
        min_bars = MIN_BARS_REQUIRED.get(timeframe, 120)
        if raw_df is None or len(raw_df) < min_bars:
            got = 0 if raw_df is None else len(raw_df)
            print(f"  ⚠️  {symbol} {timeframe}: Insufficient data (have {got} < {min_bars} bars)")
            return

        last_bar_time = raw_df.index[-1]
        now = datetime.now(timezone.utc)
        staleness = now - last_bar_time
        staleness_minutes = staleness.total_seconds() / 60

        # CRITICAL: Only generate signals on LIVE prices
        # Forex market hours: Sunday 10pm UTC - Friday 10pm UTC (24/5)

        # Check if market is closed
        if now.weekday() == 5:  # Saturday - always closed
            print(f"  ⚠️  {symbol} {timeframe}: SATURDAY - Market closed, skipping signal generation")
            return
        elif now.weekday() == 6 and now.hour < 22:  # Sunday before 10pm UTC
            print(f"  ⚠️  {symbol} {timeframe}: MARKET CLOSED - Sunday {now.hour}:00 UTC (opens 22:00)")
            return
        elif now.weekday() == 4 and now.hour >= 22:  # Friday after 10pm UTC
            print(f"  ⚠️  {symbol} {timeframe}: MARKET CLOSING - Friday {now.hour}:00 UTC, skipping signals")
            return

        # Strict staleness thresholds for LIVE trading (1-2 bar widths max)
        # These ensure signals are on current prices, not historical data
        max_allowed_minutes = {
            '5T': 15,    # 3 bars max (allows for API delays)
            '15T': 30,   # 2 bars max
            '30T': 45,   # 1.5 bars max
            '1H': 90,    # 1.5 bars max
        }.get(timeframe, 30)

        # Absolute safety limit: max 2 hours for all intraday timeframes
        absolute_max = 120
        if staleness_minutes > absolute_max:
            print(f"  ❌ {symbol} {timeframe}: Data too stale ({staleness_minutes:.0f}min > {absolute_max}min absolute limit)")
            return

        if staleness_minutes > max_allowed_minutes:
            print(f"  ⚠️  {symbol} {timeframe}: Stale data - {staleness_minutes:.0f}min old (max {max_allowed_minutes}min for LIVE trading)")
            return

        # Log exact freshness for monitoring
        print(f"  ✅ {symbol} {timeframe}: Fresh data - {staleness_minutes:.1f}min old (last bar: {last_bar_time.strftime('%H:%M UTC')})")

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

        # CRITICAL: Fetch LIVE current price for signal execution
        # Historical data is used for features/patterns, live price for entry
        print(f"  💰 Fetching LIVE current price...")
        live_price = fetch_live_price(symbol)

        if live_price is None:
            # Fallback to last historical bar if live price unavailable
            print(f"  ⚠️  Using last historical bar price as fallback")
            live_price = float(raw_df['close'].iloc[-1])
            historical_price = live_price
        else:
            historical_price = float(raw_df['close'].iloc[-1])
            price_diff = abs(live_price - historical_price)
            price_diff_pct = (price_diff / historical_price) * 100

            print(f"  📊 Historical price: ${historical_price:.2f} (from {raw_df.index[-1].strftime('%H:%M')})")
            print(f"  💵 LIVE price: ${live_price:.2f} (difference: ${price_diff:.2f}, {price_diff_pct:.2f}%)")

            # Safety check: if prices diverge too much, something is wrong
            if price_diff_pct > 5:  # More than 5% difference is suspicious
                print(f"  ❌ WARNING: Live vs historical price differs by {price_diff_pct:.2f}% - possible data issue")

        current_close = live_price  # Use LIVE price for entry
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
        # FIXED: Arguments must be in correct order (entry_price, atr, direction)
        tp_price, sl_price = calculate_tp_sl_prices(symbol, timeframe, estimated_entry, atr, signal_type)

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

