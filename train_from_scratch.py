#!/usr/bin/env python3
"""
TRAIN FROM SCRATCH - XAUUSD 15T
===============================
Uses existing feature_store data, creates proper labels, trains fresh model.
NO API DEPENDENCY - works offline!
"""

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

from market_costs import get_tp_sl
from shared_features import calculate_features


def load_data(symbol, timeframe):
    """Load OHLCV from feature_store."""
    data_path = Path(f"feature_store/{symbol}/{symbol}_{timeframe}.parquet")

    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return None

    df = pd.read_parquet(data_path)

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    print(f"✅ Loaded {len(df):,} bars from {data_path}")
    return df


def create_labels(df, symbol, timeframe):
    """Create labels with NEXT BAR OPEN entry."""
    df = df.copy()
    n = len(df)
    horizon = 40

    tp_sl = get_tp_sl(symbol, timeframe)
    tp_mult = tp_sl.tp_atr_mult
    sl_mult = tp_sl.sl_atr_mult

    print(f"📊 Using TP={tp_mult:.1f}R, SL={sl_mult:.1f}R")

    atr = df['atr14'].values
    next_opens = df['open'].shift(-1).values  # NEXT BAR OPEN!

    tp_prices = next_opens + (atr * tp_mult)
    sl_prices = next_opens - (atr * sl_mult)

    labels = np.zeros(n, dtype=int)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    for i in range(n - horizon - 1):
        if np.isnan(next_opens[i]) or np.isnan(atr[i]):
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


def train_model(df):
    """Train XGBoost binary classifier."""
    feature_cols = [col for col in df.columns
                   if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]

    X = df[feature_cols].values
    y = df['target'].values

    # Check class distribution
    unique_y, counts_y = np.unique(y, return_counts=True)
    print(f"\n📊 Class Distribution: {dict(zip(unique_y, counts_y))}")

    # Filter out class 0 (Flat) - trade only directional moves
    mask = y != 0
    X = X[mask]
    y = y[mask]

    if len(y) < 100:
        print(f"❌ Not enough samples: {len(y)}")
        return None, None

    # Remap: 1→0 (Up), 2→1 (Down)
    y = np.where(y == 1, 0, 1)

    print(f"\n🎯 Training Binary Classifier:")
    print(f"   Up (0): {(y == 0).sum():,} samples")
    print(f"   Down (1): {(y == 1).sum():,} samples")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📈 Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train
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

    print(f"\n🔥 Training XGBoost...")
    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()

    print(f"\n✅ Test Accuracy: {acc:.1%}")
    print("\n" + classification_report(y_test, y_pred, target_names=['Up', 'Down'], zero_division=0))

    return model, feature_cols


def save_model(model, features, symbol, timeframe):
    """Save model."""
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
                'directional_only': True,
                'training_script': 'train_from_scratch.py'
            }
        }, f)

    print(f"\n💾 Saved: {model_path}")
    return model_path


def main():
    symbol = 'XAUUSD'
    timeframe = '15T'

    print("\n" + "="*80)
    print(f"TRAINING FROM SCRATCH - {symbol} {timeframe}")
    print("="*80 + "\n")

    # Load data
    df = load_data(symbol, timeframe)
    if df is None:
        return 1

    # Calculate features
    print(f"\n🔧 Calculating features...")
    df = calculate_features(df)
    print(f"✅ Features calculated: {len(df):,} bars")

    # Create labels
    print(f"\n🏷️  Creating labels (next bar open entry)...")
    df = create_labels(df, symbol, timeframe)
    print(f"✅ Labels created: {len(df):,} samples")

    # Train
    model, features = train_model(df)
    if model is None:
        return 1

    # Save
    save_model(model, features, symbol, timeframe)

    print(f"\n" + "="*80)
    print(f"✅ SUCCESS - Model ready for validation!")
    print(f"="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
