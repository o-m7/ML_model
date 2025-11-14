#!/usr/bin/env python3
"""
DEMO TRAINING - Proves the system works with synthetic data
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

from market_costs import get_tp_sl
from shared_features import calculate_features


def generate_synthetic_data(n_bars=10000):
    """Generate realistic XAUUSD price data."""
    np.random.seed(42)

    # Start at 2000
    close_price = 2000.0
    prices = []

    for i in range(n_bars):
        # Random walk with trend
        change = np.random.randn() * 5 + 0.02  # Slight upward bias
        close_price += change

        # OHLC
        high = close_price + abs(np.random.randn() * 3)
        low = close_price - abs(np.random.randn() * 3)
        open_price = close_price + np.random.randn() * 2
        volume = np.random.randint(5000, 15000)

        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

    df = pd.DataFrame(prices)
    print(f"✅ Generated {len(df):,} synthetic bars (price range: {df['close'].min():.0f}-{df['close'].max():.0f})")
    return df


def create_labels(df, symbol='XAUUSD', timeframe='15T'):
    """Create labels with NEXT BAR OPEN entry."""
    n = len(df)
    horizon = 40

    tp_sl = get_tp_sl(symbol, timeframe)
    tp_mult = tp_sl.tp_atr_mult
    sl_mult = tp_sl.sl_atr_mult

    print(f"📊 Using TP={tp_mult:.1f}R, SL={sl_mult:.1f}R")

    atr = df['atr14'].values
    next_opens = df['open'].shift(-1).values

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
            atr_norm = ret * next_opens[i] / atr[i] if atr[i] > 0 else 0

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

    unique_y, counts_y = np.unique(y, return_counts=True)
    print(f"\n📊 Class Distribution: {dict(zip(unique_y, counts_y))}")

    # Filter Flat class
    mask = y != 0
    X = X[mask]
    y = y[mask]

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
    y_proba = model.predict_proba(X_test)
    acc = (y_pred == y_test).mean()

    print(f"\n✅ Test Accuracy: {acc:.1%}")

    # Check confidence distribution
    confidences = y_proba.max(axis=1)
    high_conf = (confidences >= 0.55).sum()
    print(f"📊 High confidence (≥55%): {high_conf}/{len(confidences)} = {high_conf/len(confidences)*100:.1f}%")

    print("\n" + classification_report(y_test, y_pred, target_names=['Up', 'Down'], zero_division=0))

    return model, feature_cols


def save_model(model, features):
    """Save model."""
    model_dir = Path("models_rentec/XAUUSD")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "XAUUSD_15T.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': features,
            'class_names': ['Up', 'Down'],
            'metadata': {
                'symbol': 'XAUUSD',
                'timeframe': '15T',
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'entry_method': 'next_bar_open_FIXED',
                'directional_only': True,
                'training_script': 'train_demo.py',
                'data_type': 'synthetic_demo'
            }
        }, f)

    print(f"\n💾 Saved: {model_path}")


def main():
    print("\n" + "="*80)
    print("DEMO TRAINING - XAUUSD 15T (Synthetic Data)")
    print("="*80 + "\n")

    # Generate data
    print("🎲 Generating synthetic data...")
    df = generate_synthetic_data(n_bars=10000)

    # Calculate features
    print(f"\n🔧 Calculating features...")
    df = calculate_features(df)
    print(f"✅ Features calculated: {len(df):,} bars")

    # Create labels
    print(f"\n🏷️  Creating labels (next bar open entry)...")
    df = create_labels(df)
    print(f"✅ Labels created: {len(df):,} samples")

    # Train
    model, features = train_model(df)

    # Save
    save_model(model, features)

    print(f"\n" + "="*80)
    print(f"✅ SUCCESS - Demo model created!")
    print(f"\nThis proves the training system works.")
    print(f"Now you need to run this with REAL data on your local machine.")
    print(f"="*80 + "\n")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
