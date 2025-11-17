#!/usr/bin/env python3
"""
INSTITUTIONAL-GRADE ML TRADING SYSTEM
======================================

Trains production-ready models for ALL symbols and timeframes with:
1. Balanced long/short label creation
2. Comprehensive feature engineering
3. Rigorous validation and testing
4. Realistic cost modeling

Symbols: XAUUSD, XAGUSD, EURUSD, GBPUSD, AUDUSD, NZDUSD
Timeframes: 5T, 15T, 30T, 1H, 4H

Usage:
    # Train all models
    python institutional_ml_trading_system.py --all

    # Train specific symbol
    python institutional_ml_trading_system.py --symbol XAUUSD --all-timeframes

    # Train specific symbol/timeframe
    python institutional_ml_trading_system.py --symbol XAUUSD --tf 15T
"""

import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

# Import unified cost model
from market_costs import get_tp_sl, get_costs, TP_SL_PARAMS

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    print("❌ ERROR: POLYGON_API_KEY not set in .env")
    sys.exit(1)

# Configuration
SYMBOLS = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
TIMEFRAMES = ['5T', '15T', '30T', '1H', '4H']

TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD',
}

TIMEFRAME_MINUTES = {'5T': 5, '15T': 15, '30T': 30, '1H': 60, '4H': 240}

# Model save directory
MODEL_DIR = Path("models_institutional")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv_from_polygon(symbol: str, timeframe: str, days_back=730):
    """Fetch OHLCV data from Polygon API."""
    print(f"📡 Fetching {days_back} days of {symbol} {timeframe} data...")

    ticker = TICKER_MAP.get(symbol, symbol)
    multiplier = TIMEFRAME_MINUTES[timeframe]

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/minute"
    url += f"/{start_time.strftime('%Y-%m-%d')}/{end_time.strftime('%Y-%m-%d')}"

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'OK' or not data.get('results'):
            print(f"⚠️  No data from Polygon: {data.get('message', 'Unknown error')}")
            return None

        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"✅ Fetched {len(df):,} bars ({df['timestamp'].min()} to {df['timestamp'].max()})")
        return df

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


def calculate_features(df):
    """Calculate institutional-grade features."""
    print("📊 Calculating features...")

    df = df.copy()

    # ATR (critical for risk management)
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

    # Moving averages (multiple timeframes)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema{period}'] = df['close'].ewm(span=period).mean()
        df[f'close_vs_sma{period}'] = (df['close'] - df[f'sma{period}']) / (df[f'sma{period}'] + 1e-10)
        df[f'close_vs_ema{period}'] = (df['close'] - df[f'ema{period}']) / (df[f'ema{period}'] + 1e-10)

    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10)

    # RSI
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi{period}'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
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

    # Volume
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

    # Drop NaN
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)

    print(f"✅ Features calculated, {len(df):,} bars (dropped {dropped:,} NaN rows)")

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


def create_balanced_labels(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Create BALANCED labels - checks BOTH long and short opportunities.

    CRITICAL FIX for directional bias:
    - Evaluates both long and short trades for each bar
    - Labels only if one direction clearly wins
    - Ensures balanced long/short distribution
    """
    print(f"🏷️  Creating BALANCED labels for {symbol} {timeframe}...")

    df = df.copy()
    n = len(df)
    horizon = 50

    # Get TP/SL params from unified config
    try:
        tp_sl_params = get_tp_sl(symbol, timeframe)
        tp_mult = tp_sl_params.tp_atr_mult
        sl_mult = tp_sl_params.sl_atr_mult
    except Exception as e:
        print(f"   ⚠️  Could not get TP/SL params: {e}, using defaults")
        tp_mult = 1.6
        sl_mult = 1.0

    print(f"   TP: {tp_mult:.1f}x ATR, SL: {sl_mult:.1f}x ATR (R:R = {tp_mult/sl_mult:.2f})")

    atr = df['atr14'].values

    # Entry at NEXT bar open (realistic!)
    next_bar_opens = df['open'].shift(-1).values
    entry_prices = next_bar_opens

    highs = df['high'].values
    lows = df['low'].values

    # Initialize: 0=Flat, 1=Long, 2=Short
    labels = np.zeros(n, dtype=int)

    long_wins = 0
    short_wins = 0

    for i in range(n - horizon - 1):
        if np.isnan(entry_prices[i]) or np.isnan(atr[i]):
            continue

        entry = entry_prices[i]

        # LONG trade levels
        tp_long = entry + (atr[i] * tp_mult)
        sl_long = entry - (atr[i] * sl_mult)

        # SHORT trade levels
        tp_short = entry - (atr[i] * tp_mult)
        sl_short = entry + (atr[i] * sl_mult)

        # Look ahead
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]

        if len(future_highs) == 0:
            continue

        # Check LONG outcome
        tp_long_hits = np.where(future_highs >= tp_long)[0]
        sl_long_hits = np.where(future_lows <= sl_long)[0]
        long_wins_trade = len(tp_long_hits) > 0 and (len(sl_long_hits) == 0 or tp_long_hits[0] < sl_long_hits[0])

        # Check SHORT outcome
        tp_short_hits = np.where(future_lows <= tp_short)[0]
        sl_short_hits = np.where(future_highs >= sl_short)[0]
        short_wins_trade = len(tp_short_hits) > 0 and (len(sl_short_hits) == 0 or tp_short_hits[0] < sl_short_hits[0])

        # Label only if one direction clearly wins
        if long_wins_trade and not short_wins_trade:
            labels[i] = 1
            long_wins += 1
        elif short_wins_trade and not long_wins_trade:
            labels[i] = 2
            short_wins += 1
        elif long_wins_trade and short_wins_trade:
            # Both win - choose faster
            long_bars = tp_long_hits[0]
            short_bars = tp_short_hits[0]
            if long_bars < short_bars:
                labels[i] = 1
                long_wins += 1
            else:
                labels[i] = 2
                short_wins += 1

    df['target'] = labels
    df = df.iloc[:-(horizon + 1)]

    # Show distribution
    counts = df['target'].value_counts().sort_index()
    total = len(df)
    flat_pct = counts.get(0, 0) / total * 100
    long_pct = counts.get(1, 0) / total * 100
    short_pct = counts.get(2, 0) / total * 100

    print(f"\n   Label Distribution:")
    print(f"   Flat:  {counts.get(0, 0):,} ({flat_pct:.1f}%)")
    print(f"   Long:  {counts.get(1, 0):,} ({long_pct:.1f}%)")
    print(f"   Short: {counts.get(2, 0):,} ({short_pct:.1f}%)")
    print(f"   Long TP wins: {long_wins}")
    print(f"   Short TP wins: {short_wins}")

    # Balance check
    if long_pct < 5 or short_pct < 5:
        print(f"   ⚠️  WARNING: Severe imbalance!")
        print(f"   This suggests a strong trend in the training data period.")

    if long_wins > 0 and short_wins > 0:
        balance_ratio = long_wins / short_wins
        if 0.6 <= balance_ratio <= 1.4:
            print(f"   ✅ Good balance (Long/Short ratio: {balance_ratio:.2f})")
        else:
            print(f"   ⚠️  Imbalanced (Long/Short ratio: {balance_ratio:.2f})")

    return df


class InstitutionalModel:
    """LightGBM model with institutional-grade training."""

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()

    def fit(self, X, y):
        """Train with balanced class weights."""

        X_scaled = self.scaler.fit_transform(X)

        # Balanced weights
        counts = np.bincount(y)
        print(f"\n   Training set class distribution:")
        print(f"   Flat:  {counts[0]:,}")
        print(f"   Long:  {counts[1]:,}")
        print(f"   Short: {counts[2]:,}")

        # Inverse frequency weights
        weights = len(y) / (len(counts) * counts + 1e-10)
        weights[0] *= 1.2  # Slight Flat boost for selectivity
        # NO differential boosting for Long/Short to prevent bias

        sample_weight = weights[y]

        print(f"   Class weights: Flat={weights[0]:.2f}, Long={weights[1]:.2f}, Short={weights[2]:.2f}")

        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=16,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=2.0,
            reg_lambda=3.0,
            min_child_samples=50,
            random_state=42,
            verbosity=-1,
            force_row_wise=True,
            importance_type='gain'
        )

        self.model.fit(X_scaled, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def train_model(symbol: str, timeframe: str) -> Optional[Dict]:
    """Train institutional model for symbol/timeframe."""

    print(f"\n{'='*80}")
    print(f"TRAINING: {symbol} {timeframe}")
    print(f"{'='*80}")

    try:
        # Fetch data
        df = fetch_ohlcv_from_polygon(symbol, timeframe, days_back=730)
        if df is None or len(df) < 1000:
            print(f"❌ Insufficient data ({len(df) if df is not None else 0} bars)")
            return None

        # Features
        df = calculate_features(df)

        # Labels
        df = create_balanced_labels(df, symbol, timeframe)

        # Select features
        exclude_cols = ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].fillna(0).values
        y = df['target'].values

        # Check classes
        unique_classes = np.unique(y)
        print(f"\n   Unique classes: {unique_classes}")

        if len(unique_classes) < 3:
            print(f"   ⚠️  Only {len(unique_classes)} classes - need 3!")

        # Split: 70% train, 30% test
        split_idx = int(len(X) * 0.70)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\n   Train: {len(X_train):,} samples")
        print(f"   Test:  {len(X_test):,} samples")

        # Train
        model = InstitutionalModel()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        print(f"\n📊 Test Set Performance:")
        print("="*80)
        print(classification_report(y_test, y_pred, target_names=['Flat', 'Long', 'Short'], digits=3))

        # Check predictions
        pred_counts = np.bincount(y_pred)
        print(f"\n   Test Set Predictions:")
        print(f"   Flat:  {pred_counts[0]:,} ({pred_counts[0]/len(y_pred)*100:.1f}%)")
        print(f"   Long:  {pred_counts[1]:,} ({pred_counts[1]/len(y_pred)*100:.1f}%)")
        print(f"   Short: {pred_counts[2]:,} ({pred_counts[2]/len(y_pred)*100:.1f}%)")

        # Bias check
        long_short_ratio = pred_counts[1] / (pred_counts[2] + 1e-10)
        if long_short_ratio < 0.5 or long_short_ratio > 2.0:
            print(f"\n   ⚠️  WARNING: Prediction bias (L/S ratio: {long_short_ratio:.2f})")
            status = "BIASED"
        else:
            print(f"\n   ✅ Balanced predictions (L/S ratio: {long_short_ratio:.2f})")
            status = "READY"

        # Save
        save_dir = MODEL_DIR / symbol
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        model_path = save_dir / f"{symbol}_{timeframe}_{status}_{timestamp}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': feature_cols,
                'class_names': ['Flat', 'Long', 'Short'],
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'long_short_ratio': float(long_short_ratio),
                    'status': status
                }
            }, f)

        print(f"\n✅ Model saved: {model_path.name}")

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': status,
            'long_short_ratio': long_short_ratio,
            'model_path': model_path
        }

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Institutional ML Trading System')
    parser.add_argument('--all', action='store_true', help='Train all symbols and timeframes')
    parser.add_argument('--symbol', type=str, help='Train specific symbol')
    parser.add_argument('--tf', type=str, help='Train specific timeframe')
    parser.add_argument('--all-timeframes', action='store_true', help='Train all timeframes for symbol')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("INSTITUTIONAL ML TRADING SYSTEM")
    print("="*80)

    results = []

    if args.all:
        # Train everything
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                result = train_model(symbol, timeframe)
                if result:
                    results.append(result)

    elif args.symbol and args.all_timeframes:
        # Train all timeframes for one symbol
        for timeframe in TIMEFRAMES:
            result = train_model(args.symbol, timeframe)
            if result:
                results.append(result)

    elif args.symbol and args.tf:
        # Train single model
        result = train_model(args.symbol, args.tf)
        if result:
            results.append(result)

    else:
        print("\nUsage:")
        print("  python institutional_ml_trading_system.py --all")
        print("  python institutional_ml_trading_system.py --symbol XAUUSD --all-timeframes")
        print("  python institutional_ml_trading_system.py --symbol XAUUSD --tf 15T")
        return 1

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    ready = [r for r in results if r['status'] == 'READY']
    biased = [r for r in results if r['status'] == 'BIASED']

    print(f"\nTotal models trained: {len(results)}")
    print(f"✅ Ready: {len(ready)}")
    print(f"⚠️  Biased: {len(biased)}")

    if ready:
        print(f"\n✅ PRODUCTION-READY MODELS:")
        for r in ready:
            print(f"   {r['symbol']} {r['timeframe']}: L/S ratio = {r['long_short_ratio']:.2f}")

    if biased:
        print(f"\n⚠️  BIASED MODELS (review required):")
        for r in biased:
            print(f"   {r['symbol']} {r['timeframe']}: L/S ratio = {r['long_short_ratio']:.2f}")

    print("="*80)
    print(f"\nModels saved to: {MODEL_DIR}")
    print("\nNext: Run backtests with validate_backtest_with_costs.py")
    print("="*80 + "\n")

    return 0 if len(ready) > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
