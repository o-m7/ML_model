#!/usr/bin/env python3
"""
COMPLETE MODEL RETRAINING SYSTEM - ALL SYMBOLS & TIMEFRAMES
============================================================

Fetches fresh data from Polygon, creates FIXED labels (next bar open),
and trains models from scratch. No dependencies on old failed models.

Usage:
    python retrain_all_models.py --symbol XAUUSD --tf 15T
    python retrain_all_models.py --all
"""

import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

from market_costs import get_tp_sl

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY or POLYGON_API_KEY == 'your_polygon_api_key_here':
    print("❌ ERROR: Set POLYGON_API_KEY in .env file")
    sys.exit(1)

# All symbols and timeframes
SYMBOLS = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
TIMEFRAMES = ['5T', '15T', '30T', '1H', '4H']

TICKER_MAP = {
    'XAUUSD': 'C:XAUUSD',
    'XAGUSD': 'C:XAGUSD',
    'EURUSD': 'C:EURUSD',
    'GBPUSD': 'C:GBPUSD',
    'AUDUSD': 'C:AUDUSD',
    'NZDUSD': 'C:NZDUSD'
}

TF_MINUTES = {
    '5T': 5,
    '15T': 15,
    '30T': 30,
    '1H': 60,
    '4H': 240
}


def fetch_data(symbol, timeframe, days_back=365):
    """Fetch OHLCV from Polygon."""
    ticker = TICKER_MAP[symbol]
    minutes = TF_MINUTES[timeframe]

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{minutes}/minute"
    url += f"/{start_time.strftime('%Y-%m-%d')}/{end_time.strftime('%Y-%m-%d')}"

    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}

    print(f"  📡 Fetching {symbol} {timeframe}...")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'OK' or not data.get('results'):
            print(f"  ❌ No data from Polygon")
            return None

        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"  ✅ Got {len(df):,} bars")
        return df

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def add_features(df):
    """Calculate technical indicators."""
    df = df.copy()

    # ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(window=14).mean()

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages
    for period in [10, 20, 50]:
        df[f'sma{period}'] = df['close'].rolling(window=period).mean()
        df[f'close_vs_sma{period}'] = (df['close'] - df[f'sma{period}']) / df[f'sma{period}']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))

    # Bollinger
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df = df.dropna().reset_index(drop=True)
    return df


def create_labels(df, symbol, timeframe):
    """Create labels with NEXT BAR OPEN entry (FIXED!)."""
    df = df.copy()
    n = len(df)
    horizon = 40

    tp_sl = get_tp_sl(symbol, timeframe)
    tp_mult = tp_sl.tp_atr_mult
    sl_mult = tp_sl.sl_atr_mult

    atr = df['atr14'].values
    next_opens = df['open'].shift(-1).values  # NEXT BAR OPEN!

    tp_prices = next_opens + (atr * tp_mult)
    sl_prices = next_opens - (atr * sl_mult)

    labels = np.zeros(n, dtype=int)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    for i in range(n - horizon - 1):
        if np.isnan(next_opens[i]):
            continue

        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]
        future_closes = closes[i+1:i+1+horizon]

        if len(future_highs) == 0:
            continue

        tp_hits = np.where(future_highs >= tp_prices[i])[0]
        sl_hits = np.where(future_lows <= sl_prices[i])[0]

        if len(tp_hits) > 0 and len(sl_hits) > 0:
            labels[i] = 1 if tp_hits[0] < sl_hits[0] else 2
        elif len(tp_hits) > 0:
            labels[i] = 1
        elif len(sl_hits) > 0:
            labels[i] = 2
        else:
            final = future_closes[-1]
            ret = (final - next_opens[i]) / next_opens[i]
            atr_norm = ret * next_opens[i] / atr[i]

            if atr_norm >= (tp_mult * 0.85):
                labels[i] = 1
            elif atr_norm <= -(sl_mult * 0.85):
                labels[i] = 2

    df['target'] = labels
    df = df.iloc[:-(horizon + 1)]

    return df


def train_xgboost(df, symbol, timeframe):
    """Train XGBoost model with stratified split."""
    feature_cols = [col for col in df.columns
                   if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]

    X = df[feature_cols].values
    y = df['target'].values

    # Check class distribution
    unique_y, counts_y = np.unique(y, return_counts=True)
    print(f"  📊 Classes: {dict(zip(unique_y, counts_y))}")

    # Filter out class 0 (Flat) for better performance
    # Trade only clear directional moves
    mask = y != 0
    X = X[mask]
    y = y[mask]

    if len(y) < 100:
        print(f"  ❌ Not enough samples after filtering: {len(y)}")
        return None, None

    # Remap: 1→0 (Up), 2→1 (Down)
    y = np.where(y == 1, 0, 1)

    print(f"  🎯 Training on directional signals only")
    print(f"     Up: {(y == 0).sum()}, Down: {(y == 1).sum()}")

    # Stratified split to ensure both classes in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  📈 Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train binary classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()

    print(f"  ✅ Test Accuracy: {acc:.1%}")
    print(classification_report(y_test, y_pred, target_names=['Up', 'Down'], zero_division=0))

    return model, feature_cols


def save_model(model, features, symbol, timeframe):
    """Save model to models_rentec."""
    model_dir = Path("models_rentec") / symbol
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{symbol}_{timeframe}.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': features,
            'class_names': ['Up', 'Down'],
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'entry_method': 'next_bar_open_FIXED',
                'directional_only': True
            }
        }, f)

    print(f"  💾 Saved: {model_path}")
    return model_path


def train_single(symbol, timeframe):
    """Train a single model."""
    print(f"\n{'='*80}")
    print(f"{symbol} {timeframe}")
    print(f"{'='*80}")

    # Fetch data
    df = fetch_data(symbol, timeframe, days_back=365)
    if df is None or len(df) < 1000:
        print(f"  ❌ Insufficient data")
        return False

    # Add features
    df = add_features(df)

    # Create labels
    df = create_labels(df, symbol, timeframe)

    # Train
    model, features = train_xgboost(df, symbol, timeframe)
    if model is None:
        return False

    # Save
    save_model(model, features, symbol, timeframe)

    print(f"  ✅ SUCCESS")
    return True


def main():
    parser = argparse.ArgumentParser(description='Retrain models with FIXED labels')
    parser.add_argument('--symbol', type=str, help='Single symbol to train')
    parser.add_argument('--tf', type=str, help='Single timeframe to train')
    parser.add_argument('--all', action='store_true', help='Train all symbols/timeframes')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MODEL RETRAINING - FIXED LABELS (NEXT BAR OPEN)")
    print("="*80 + "\n")

    if args.all:
        success = 0
        total = 0
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                total += 1
                if train_single(symbol, timeframe):
                    success += 1

        print(f"\n{'='*80}")
        print(f"COMPLETE: {success}/{total} models trained successfully")
        print(f"{'='*80}\n")

    elif args.symbol and args.tf:
        if args.symbol not in SYMBOLS:
            print(f"❌ Invalid symbol: {args.symbol}")
            print(f"   Available: {SYMBOLS}")
            return 1

        if args.tf not in TIMEFRAMES:
            print(f"❌ Invalid timeframe: {args.tf}")
            print(f"   Available: {TIMEFRAMES}")
            return 1

        success = train_single(args.symbol, args.tf)
        return 0 if success else 1

    else:
        print("Usage:")
        print("  python retrain_all_models.py --symbol XAUUSD --tf 15T")
        print("  python retrain_all_models.py --all")
        return 1


if __name__ == '__main__':
    sys.exit(main())
