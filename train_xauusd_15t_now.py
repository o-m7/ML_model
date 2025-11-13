#!/usr/bin/env python3
"""
QUICK XAUUSD 15T TRAINING - FETCHES DATA FROM POLYGON
======================================================

No dependencies on feature_store or external pipelines.
Fetches data directly from Polygon API and trains immediately.

Usage:
    python train_xauusd_15t_now.py
"""

import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Import our fixed modules
from market_costs import get_tp_sl

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    print("❌ ERROR: POLYGON_API_KEY not set in .env")
    sys.exit(1)

SYMBOL = 'XAUUSD'
TIMEFRAME = '15T'
TICKER = 'C:XAUUSD'

print("\n" + "="*80)
print(f"XAUUSD 15T TRAINING - WITH FIXED LABELS")
print("="*80 + "\n")


def fetch_ohlcv_from_polygon(days_back=365):
    """Fetch OHLCV data from Polygon API."""
    print(f"📡 Fetching {days_back} days of data from Polygon...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    url = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/15/minute"
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
            print(f"❌ No data from Polygon: {data.get('message')}")
            return None

        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"✅ Fetched {len(df):,} bars")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


def calculate_features(df):
    """Calculate technical indicators."""
    print("\n📊 Calculating features...")

    df = df.copy()

    # ATR (critical for labeling)
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(window=14).mean()

    # Price-based features
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

    # Bollinger Bands
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    print(f"✅ Calculated features, {len(df):,} bars remaining after dropna()")

    return df


def create_labels(df):
    """
    Create labels using NEXT BAR OPEN (FIXED!)
    """
    print("\n🏷️  Creating labels with NEXT BAR OPEN entry...")

    df = df.copy()
    n = len(df)
    horizon = 40

    tp_sl_params = get_tp_sl(SYMBOL, TIMEFRAME)
    tp_mult = tp_sl_params.tp_atr_mult
    sl_mult = tp_sl_params.sl_atr_mult

    print(f"   TP: {tp_mult}x ATR, SL: {sl_mult}x ATR")

    atr = df['atr14'].values

    # CRITICAL FIX: Entry at NEXT bar open!
    next_bar_opens = df['open'].shift(-1).values
    entry_prices = next_bar_opens

    tp_prices = entry_prices + (atr * tp_mult)
    sl_prices = entry_prices - (atr * sl_mult)

    labels = np.zeros(n, dtype=int)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    for i in range(n - horizon - 1):
        if np.isnan(entry_prices[i]):
            continue

        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]
        future_closes = closes[i+1:i+1+horizon]

        if len(future_highs) == 0:
            continue

        tp_hits = np.where(future_highs >= tp_prices[i])[0]
        sl_hits = np.where(future_lows <= sl_prices[i])[0]

        tp_hit = len(tp_hits) > 0
        sl_hit = len(sl_hits) > 0

        if tp_hit and sl_hit:
            labels[i] = 1 if tp_hits[0] < sl_hits[0] else 2
        elif tp_hit:
            labels[i] = 1  # Up
        elif sl_hit:
            labels[i] = 2  # Down
        else:
            # Check final price
            final_price = future_closes[-1]
            ret = (final_price - entry_prices[i]) / entry_prices[i]
            atr_normalized_ret = ret * entry_prices[i] / atr[i]

            if atr_normalized_ret >= (tp_mult * 0.85):
                labels[i] = 1
            elif atr_normalized_ret <= -(sl_mult * 0.85):
                labels[i] = 2
            else:
                labels[i] = 0  # Flat

    df['target'] = labels
    df = df.iloc[:-(horizon + 1)]

    # Show distribution
    counts = df['target'].value_counts()
    total = len(df)
    print(f"   Flat: {counts.get(0, 0):,} ({counts.get(0, 0)/total*100:.1f}%)")
    print(f"   Up:   {counts.get(1, 0):,} ({counts.get(1, 0)/total*100:.1f}%)")
    print(f"   Down: {counts.get(2, 0):,} ({counts.get(2, 0)/total*100:.1f}%)")

    return df


def train_model(df):
    """Train XGBoost model."""
    print("\n🤖 Training XGBoost model...")

    # Remove non-feature columns
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]

    X = df[feature_cols].values
    y = df['target'].values

    # Check unique classes
    unique_classes = np.unique(y)
    print(f"   Unique classes in data: {unique_classes}")

    # If we don't have class 0 (Flat), remap classes to be consecutive from 0
    if 0 not in unique_classes:
        print(f"   ⚠️  No Flat (0) labels found. Remapping classes...")
        # Map: 1 -> 0 (Up), 2 -> 1 (Down)
        y = np.where(y == 1, 0, 1)
        class_names = ['Up', 'Down']
        print(f"   Remapped to binary: Up=0, Down=1")
    else:
        class_names = ['Flat', 'Up', 'Down']

    # Split: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")

    # Train
    num_classes = len(np.unique(y_train))

    if num_classes == 2:
        # Binary classification
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
    else:
        # Multi-class classification
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
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
    y_proba = model.predict_proba(X_test)

    print("\n📊 Test Set Performance:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save model
    model_dir = Path("models_rentec") / SYMBOL
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{SYMBOL}_{TIMEFRAME}.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols,
            'class_names': class_names,
            'metadata': {
                'symbol': SYMBOL,
                'timeframe': TIMEFRAME,
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_classes': num_classes,
                'entry_method': 'next_bar_open_FIXED'
            }
        }, f)

    print(f"\n✅ Model saved to: {model_path}")

    return model, feature_cols


def main():
    # Fetch data
    df = fetch_ohlcv_from_polygon(days_back=365)
    if df is None or len(df) < 1000:
        print("❌ Insufficient data")
        return 1

    # Calculate features
    df = calculate_features(df)

    # Create labels
    df = create_labels(df)

    # Train
    model, features = train_model(df)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run: python validate_backtest_with_costs.py --symbol XAUUSD --tf 15T")
    print("  2. Expected: WR 55-65%, PF 1.3-1.8")
    print("\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
