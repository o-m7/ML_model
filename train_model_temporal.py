#!/usr/bin/env python3
"""
TEMPORAL MODEL TRAINING - Proper Time-Based Split
==================================================
Trains models using CHRONOLOGICAL train/test split (no data leakage).

Train on HISTORICAL data (first 80% chronologically)
Test on RECENT data (last 20% chronologically)

This is the ONLY correct way to train time series models.

Usage:
    python train_model_temporal.py --symbol XAUUSD --tf 15T
    python train_model_temporal.py --symbol XAGUSD --tf 30T
"""

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import xgboost as xgb

from live_feature_utils import build_feature_frame


def create_simple_labels(df, lookback=10):
    """
    Simple future returns labeling - SYMMETRIC and BALANCED.

    Label 1 (Up): Future price goes up significantly
    Label 0 (Down): Future price goes down significantly
    """
    df = df.copy()

    # Calculate future returns over next N bars
    future_returns = df['close'].shift(-lookback) / df['close'] - 1

    # Use median absolute return as threshold for "significant" move
    abs_returns = future_returns.abs()
    threshold = abs_returns.quantile(0.60)  # Top 40% of moves

    print(f"📊 Return threshold for significant moves: {threshold*100:.2f}%")

    # Create labels
    labels = np.zeros(len(df), dtype=int)

    # Up: significant positive return
    up_mask = future_returns >= threshold
    labels[up_mask] = 1

    # Down: significant negative return
    down_mask = future_returns <= -threshold
    labels[down_mask] = 0

    # Remove bars with no clear direction (small moves)
    neutral_mask = (future_returns > -threshold) & (future_returns < threshold)

    print(f"   Up samples: {up_mask.sum():,} ({up_mask.sum()/len(df)*100:.1f}%)")
    print(f"   Down samples: {down_mask.sum():,} ({down_mask.sum()/len(df)*100:.1f}%)")
    print(f"   Neutral (removed): {neutral_mask.sum():,} ({neutral_mask.sum()/len(df)*100:.1f}%)")

    df['target'] = labels
    df['future_return'] = future_returns

    # Remove last lookback bars (no future data) and neutral bars
    df = df.iloc[:-lookback]
    df = df[~neutral_mask[:-lookback]].copy()

    # Ensure index is preserved for temporal splitting
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"   ⚠️  Warning: Index is not DatetimeIndex, temporal split may be incorrect")

    return df


def temporal_train_test_split(df, test_size=0.2):
    """
    TEMPORAL split - train on FIRST 80%, test on LAST 20%.

    This is the ONLY correct way for time series.
    """
    # Data should already be sorted by timestamp (index)
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\n⏰ TEMPORAL SPLIT:")
    print(f"   Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df):,} bars)")
    print(f"   Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df):,} bars)")

    return train_df, test_df


def train_model(train_df, test_df):
    """Train XGBoost binary classifier with temporal split."""
    feature_cols = [col for col in train_df.columns
                   if col not in ['target', 'future_return', 'open', 'high', 'low', 'close', 'volume']]

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    # Check balance
    unique_y, counts_y = np.unique(y_train, return_counts=True)
    print(f"\n📊 Class Balance (Train):")
    for cls, count in zip(unique_y, counts_y):
        class_name = 'Up' if cls == 1 else 'Down'
        print(f"   {class_name} ({cls}): {count:,} ({count/len(y_train)*100:.1f}%)")

    # Train with scale_pos_weight to handle any imbalance
    n_down = (y_train == 0).sum()
    n_up = (y_train == 1).sum()
    scale_pos_weight = n_down / n_up if n_up > 0 else 1.0

    print(f"\n📈 Using scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    print(f"\n🔥 Training XGBoost on HISTORICAL data...")
    model.fit(X_train, y_train, verbose=False)

    # Evaluate on RECENT data (temporal test set)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = (y_pred == y_test).mean()

    print(f"\n✅ Test Accuracy (on RECENT data): {acc:.1%}")

    # Check selectivity
    confidences = y_proba.max(axis=1)
    high_conf_50 = (confidences >= 0.50).sum()
    high_conf_55 = (confidences >= 0.55).sum()
    high_conf_60 = (confidences >= 0.60).sum()
    high_conf_70 = (confidences >= 0.70).sum()

    print(f"\n📊 Confidence Distribution (Test Set):")
    print(f"   ≥50%: {high_conf_50}/{len(confidences)} ({high_conf_50/len(confidences)*100:.1f}%)")
    print(f"   ≥55%: {high_conf_55}/{len(confidences)} ({high_conf_55/len(confidences)*100:.1f}%)")
    print(f"   ≥60%: {high_conf_60}/{len(confidences)} ({high_conf_60/len(confidences)*100:.1f}%)")
    print(f"   ≥70%: {high_conf_70}/{len(confidences)} ({high_conf_70/len(confidences)*100:.1f}%)")

    if high_conf_50 / len(confidences) > 0.90:
        print(f"\n⚠️  Model may overtrade ({high_conf_50/len(confidences)*100:.0f}% ≥50% confidence)")
    elif high_conf_50 / len(confidences) < 0.20:
        print(f"\n⚠️  Model may undertrade ({high_conf_50/len(confidences)*100:.0f}% ≥50% confidence)")
    else:
        print(f"\n✅ Good selectivity: {high_conf_50/len(confidences)*100:.1f}% ≥50% confidence")

    # Check class balance in predictions
    pred_up = (y_pred == 1).sum()
    pred_down = (y_pred == 0).sum()
    print(f"\n📊 Prediction Balance (Test Set):")
    print(f"   Predicted Up: {pred_up} ({pred_up/len(y_pred)*100:.1f}%)")
    print(f"   Predicted Down: {pred_down} ({pred_down/len(y_pred)*100:.1f}%)")

    print("\n" + classification_report(y_test, y_pred, target_names=['Down', 'Up'], zero_division=0))

    # CRITICAL: Check if model generalizes
    if acc < 0.52:
        print(f"\n⚠️  WARNING: Test accuracy {acc:.1%} is close to random (50%)")
        print(f"   Model may not generalize well to unseen data")
    elif acc >= 0.55:
        print(f"\n✅ GOOD: Test accuracy {acc:.1%} shows model generalizes")

    return model, feature_cols


def save_model(model, features, symbol, timeframe, train_period, test_period, test_acc):
    """Save model with metadata."""
    model_dir = Path(f"models_rentec/{symbol}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{symbol}_{timeframe}.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': features,
            'class_names': ['Down', 'Up'],  # 0=Down, 1=Up
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'train_period': train_period,
                'test_period': test_period,
                'test_accuracy': float(test_acc),
                'split_method': 'temporal_80_20',
                'labeling_method': 'simple_future_returns',
                'training_script': 'train_model_temporal.py'
            }
        }, f)

    print(f"\n💾 Saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train model with TEMPORAL split (no data leakage)')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol (XAUUSD, XAGUSD, etc.)')
    parser.add_argument('--tf', type=str, required=True, help='Timeframe (5T, 15T, 30T, 1H)')
    parser.add_argument('--lookback', type=int, default=10, help='Future return lookback period (default: 10)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2 = 20%%)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"TEMPORAL MODEL TRAINING - {args.symbol} {args.tf}")
    print("="*80 + "\n")

    # Load data
    data_path = Path(f"feature_store/{args.symbol}/{args.symbol}_{args.tf}.parquet")

    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        print(f"\nMake sure you have:")
        print(f"  1. Downloaded data for {args.symbol} {args.tf}")
        print(f"  2. Saved it to feature_store/{args.symbol}/{args.symbol}_{args.tf}.parquet")
        return 1

    print("📊 Loading data...")
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Calculate features (using SAME function as live trading and backtesting)
    print(f"\n🔧 Calculating features (using live_feature_utils)...")
    df = build_feature_frame(df)
    print(f"✅ Features calculated: {len(df):,} bars with {len(df.columns)} columns")

    # Create labels
    print(f"\n🏷️  Creating labels (simple future returns, lookback={args.lookback})...")
    df = create_simple_labels(df, lookback=args.lookback)
    print(f"✅ Labels created: {len(df):,} samples")

    # TEMPORAL split
    train_df, test_df = temporal_train_test_split(df, test_size=args.test_size)

    # Train
    model, features = train_model(train_df, test_df)

    # Calculate test accuracy for metadata
    X_test = test_df[features].values
    y_test = test_df['target'].values
    y_pred = model.predict(X_test)
    test_acc = (y_pred == y_test).mean()

    # Save with metadata
    train_period = f"{train_df.index[0]} to {train_df.index[-1]}"
    test_period = f"{test_df.index[0]} to {test_df.index[-1]}"
    save_model(model, features, args.symbol, args.tf, train_period, test_period, test_acc)

    print(f"\n" + "="*80)
    print(f"✅ SUCCESS - {args.symbol} {args.tf} model trained with TEMPORAL split!")
    print(f"\n📊 Summary:")
    print(f"   Trained on: {train_period}")
    print(f"   Tested on:  {test_period}")
    print(f"   Test Accuracy: {test_acc:.1%}")
    print(f"\n⚠️  IMPORTANT: Backtest should use data from AFTER the test period")
    print(f"   to avoid data leakage and get realistic performance estimates.")
    print(f"\nValidate with:")
    print(f"  python run_model_backtest.py --symbol {args.symbol} --timeframe {args.tf}")
    print(f"="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
