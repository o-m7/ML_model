#!/usr/bin/env python3
"""
FIXED MULTI-TIMEFRAME TRAINING SYSTEM - USES LOCAL DATA
=========================================================

Trains models using LOCAL data from feature_store/ (NO API calls)

Fixes for poor model performance:
1. Balanced long/short label creation
2. Improved feature engineering
3. Better model parameters to prevent directional bias
4. Uses existing local parquet files

Trains: 5T, 15T, 30T, 1H for XAUUSD (whichever timeframes exist locally)
"""

import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

# Import unified cost model and shared features
from market_costs import get_tp_sl
from shared_features import calculate_features

SYMBOL = 'XAUUSD'
TIMEFRAMES = ['5T', '15T', '30T', '1H', '4H']
FEATURE_STORE = Path("feature_store")
MODEL_DIR = Path("models_institutional")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print(f"FIXED MULTI-TIMEFRAME TRAINING - {SYMBOL} (LOCAL DATA)")
print("="*80 + "\n")


def load_local_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load data from local feature_store."""
    data_path = FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"

    if not data_path.exists():
        print(f"❌ No local data found: {data_path}")
        return None

    print(f"📁 Loading local data: {data_path.name}")

    df = pd.read_parquet(data_path)

    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            print(f"❌ No timestamp column found")
            return None

    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✅ Loaded {len(df):,} bars ({df['timestamp'].min()} to {df['timestamp'].max()})")

    # Recalculate features to ensure consistency
    print("🔧 Recalculating features for consistency...")
    df = calculate_features(df)

    return df


def create_balanced_labels(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Create BALANCED labels using NEXT BAR OPEN.

    CRITICAL FIX for directional bias:
    - Checks BOTH long and short opportunities
    - Labels only if one direction clearly wins
    - Ensures balanced long/short distribution
    """
    print(f"\n🏷️  Creating BALANCED labels for {symbol} {timeframe}...")

    df = df.copy()
    n = len(df)
    horizon = 50  # Look ahead 50 bars

    tp_sl_params = get_tp_sl(symbol, timeframe)
    tp_mult = tp_sl_params.tp_atr_mult
    sl_mult = tp_sl_params.sl_atr_mult

    print(f"   TP: {tp_mult:.1f}x ATR, SL: {sl_mult:.1f}x ATR (R:R = {tp_mult/sl_mult:.2f})")

    atr = df['atr14'].values

    # CRITICAL FIX: Entry at NEXT bar open!
    next_bar_opens = df['open'].shift(-1).values
    entry_prices = next_bar_opens

    highs = df['high'].values
    lows = df['low'].values

    # Initialize labels: 0=Flat, 1=Long, 2=Short
    labels = np.zeros(n, dtype=int)

    # Track statistics
    long_wins = 0
    short_wins = 0

    for i in range(n - horizon - 1):
        if np.isnan(entry_prices[i]) or np.isnan(atr[i]):
            continue

        entry = entry_prices[i]

        # Define TP/SL for LONG
        tp_long = entry + (atr[i] * tp_mult)
        sl_long = entry - (atr[i] * sl_mult)

        # Define TP/SL for SHORT
        tp_short = entry - (atr[i] * tp_mult)
        sl_short = entry + (atr[i] * sl_mult)

        # Look ahead
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]

        if len(future_highs) == 0:
            continue

        # Check LONG trade outcome
        tp_long_hits = np.where(future_highs >= tp_long)[0]
        sl_long_hits = np.where(future_lows <= sl_long)[0]

        long_wins_trade = len(tp_long_hits) > 0 and (len(sl_long_hits) == 0 or tp_long_hits[0] < sl_long_hits[0])

        # Check SHORT trade outcome
        tp_short_hits = np.where(future_lows <= tp_short)[0]
        sl_short_hits = np.where(future_highs >= sl_short)[0]

        short_wins_trade = len(tp_short_hits) > 0 and (len(sl_short_hits) == 0 or tp_short_hits[0] < sl_short_hits[0])

        # Label logic: Only label if ONE direction clearly wins
        if long_wins_trade and not short_wins_trade:
            labels[i] = 1  # Long
            long_wins += 1
        elif short_wins_trade and not long_wins_trade:
            labels[i] = 2  # Short
            short_wins += 1
        elif long_wins_trade and short_wins_trade:
            # Both directions win - choose the faster one
            long_bars = tp_long_hits[0]
            short_bars = tp_short_hits[0]
            if long_bars < short_bars:
                labels[i] = 1
                long_wins += 1
            else:
                labels[i] = 2
                short_wins += 1
        # else: labels[i] = 0 (Flat) - no clear winner

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
    print(f"\n   Balance Check:")
    print(f"   Long TP wins: {long_wins}")
    print(f"   Short TP wins: {short_wins}")

    # Check for severe imbalance
    if long_pct < 5 or short_pct < 5:
        print(f"   ⚠️  WARNING: Severe imbalance detected!")
        print(f"   This may indicate a strong trending period in the data.")

    if long_wins > 0 and short_wins > 0:
        balance_ratio = long_wins / short_wins
        if 0.6 <= balance_ratio <= 1.4:
            print(f"   ✅ Good balance (Long/Short ratio: {balance_ratio:.2f})")
        else:
            print(f"   ⚠️  Imbalanced (Long/Short ratio: {balance_ratio:.2f})")

    return df


class BalancedModel:
    """LightGBM model with balanced training to prevent directional bias."""

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()

    def fit(self, X, y):
        """Train with carefully balanced class weights."""

        X_scaled = self.scaler.fit_transform(X)

        # Calculate balanced weights
        counts = np.bincount(y)
        print(f"\n   Class distribution in training set:")
        print(f"   Class 0 (Flat): {counts[0]:,}")
        print(f"   Class 1 (Long): {counts[1]:,}")
        print(f"   Class 2 (Short): {counts[2]:,}")

        # Balanced weights (inverse of frequency)
        weights = len(y) / (len(counts) * counts + 1e-10)

        # CRITICAL: Don't over-boost any class
        # Apply gentle boosting to minority classes only
        weights[0] *= 1.2  # Slight Flat boost (encourage selectivity)
        # Do NOT boost Long or Short differently to avoid bias

        sample_weight = weights[y]

        print(f"\n   Class weights: Flat={weights[0]:.2f}, Long={weights[1]:.2f}, Short={weights[2]:.2f}")

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


def train_model(df, symbol, timeframe):
    """Train model with comprehensive evaluation."""
    print(f"\n🤖 Training model for {symbol} {timeframe}...")

    # Remove non-feature columns
    exclude_cols = ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].fillna(0).values
    y = df['target'].values

    # Check unique classes
    unique_classes = np.unique(y)
    print(f"\n   Unique classes in data: {unique_classes}")

    if len(unique_classes) < 3:
        print(f"   ⚠️  WARNING: Only {len(unique_classes)} classes found!")
        print(f"   Expected 3 classes (Flat, Long, Short)")

    # Split: 70% train, 30% test (more test data for validation)
    split_idx = int(len(X) * 0.70)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")

    # Train
    model = BalancedModel()
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"\n📊 Test Set Performance:")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=['Flat', 'Long', 'Short'], digits=3))

    # Check prediction distribution
    pred_counts = np.bincount(y_pred)
    print(f"\n   Prediction Distribution on Test Set:")
    print(f"   Flat:  {pred_counts[0]:,} ({pred_counts[0]/len(y_pred)*100:.1f}%)")
    print(f"   Long:  {pred_counts[1]:,} ({pred_counts[1]/len(y_pred)*100:.1f}%)")
    print(f"   Short: {pred_counts[2]:,} ({pred_counts[2]/len(y_pred)*100:.1f}%)")

    # Check for prediction bias
    long_short_ratio = pred_counts[1] / (pred_counts[2] + 1e-10)
    if long_short_ratio < 0.5 or long_short_ratio > 2.0:
        print(f"\n   ⚠️  WARNING: Prediction bias detected!")
        print(f"   Long/Short ratio: {long_short_ratio:.2f} (should be ~1.0)")
        status = "REVIEW"
    else:
        print(f"\n   ✅ Predictions well-balanced (Long/Short ratio: {long_short_ratio:.2f})")
        status = "READY"

    # Save model
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
                'num_classes': 3,
                'entry_method': 'next_bar_open_FIXED',
                'long_short_ratio': float(long_short_ratio),
                'data_source': 'local_feature_store'
            }
        }, f)

    print(f"\n✅ Model saved to: {model_path}")

    return model, feature_cols, model_path, status


def main():
    """Train models for all available local timeframes."""

    results = []

    for timeframe in TIMEFRAMES:
        print("\n" + "="*80)
        print(f"TRAINING: {SYMBOL} {timeframe}")
        print("="*80)

        try:
            # Load local data
            df = load_local_data(SYMBOL, timeframe)
            if df is None:
                print(f"⚠️  Skipping {timeframe} (no local data)")
                continue

            if len(df) < 500:
                print(f"❌ Insufficient data for {timeframe} ({len(df)} bars)")
                continue

            # Create balanced labels
            df = create_balanced_labels(df, SYMBOL, timeframe)

            # Train model
            model, features, model_path, status = train_model(df, SYMBOL, timeframe)

            results.append({
                'timeframe': timeframe,
                'success': True,
                'status': status,
                'model_path': model_path
            })

            print(f"\n{'='*80}")
            print(f"✅ {timeframe} TRAINING COMPLETE - {status}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n❌ ERROR training {timeframe}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'timeframe': timeframe,
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    ready = [r for r in results if r['success'] and r.get('status') == 'READY']
    review = [r for r in results if r['success'] and r.get('status') == 'REVIEW']

    for r in results:
        tf = r['timeframe']
        if r['success']:
            status = "✅ READY" if r.get('status') == 'READY' else "⚠️  REVIEW"
            print(f"{tf:6s}: {status}")
        else:
            print(f"{tf:6s}: ❌ FAILED - {r.get('error', 'Unknown error')}")

    print("\n" + "="*80)
    print(f"Ready models: {len(ready)}/{len(results)}")
    if review:
        print(f"Review needed: {len(review)} models")
    print("="*80)
    print("\nNext steps:")
    print("  1. Validate: python validate_all_models.py --symbol XAUUSD")
    print("  2. Backtest: python validate_backtest_with_costs.py --symbol XAUUSD --tf 15T")
    print("\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
