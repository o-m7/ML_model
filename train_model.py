#!/usr/bin/env python3
"""
TRAIN MODEL - Any Symbol/Timeframe
===================================
Uses symmetric balanced labeling to train models for any symbol and timeframe.

Usage:
    python train_model.py --symbol XAUUSD --tf 15T
    python train_model.py --symbol XAGUSD --tf 1H
"""

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

from shared_features import calculate_features


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
    df = df[~neutral_mask[:-lookback]]

    return df


def train_model(df):
    """Train XGBoost binary classifier."""
    feature_cols = [col for col in df.columns
                   if col not in ['target', 'future_return', 'open', 'high', 'low', 'close', 'volume']]

    X = df[feature_cols].values
    y = df['target'].values

    # Check balance
    unique_y, counts_y = np.unique(y, return_counts=True)
    print(f"\n📊 Class Balance:")
    for cls, count in zip(unique_y, counts_y):
        class_name = 'Up' if cls == 1 else 'Down'
        print(f"   {class_name} ({cls}): {count:,} ({count/len(y)*100:.1f}%)")

    # Should be roughly balanced (40-60% each)
    up_pct = (y == 1).sum() / len(y)
    if up_pct < 0.4 or up_pct > 0.6:
        print(f"\n⚠️  WARNING: Classes are imbalanced ({up_pct*100:.1f}% Up)")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📈 Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train with scale_pos_weight to handle any remaining imbalance
    n_down = (y_train == 0).sum()
    n_up = (y_train == 1).sum()
    scale_pos_weight = n_down / n_up if n_up > 0 else 1.0

    print(f"   Using scale_pos_weight: {scale_pos_weight:.2f}")

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

    print(f"\n🔥 Training XGBoost...")
    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = (y_pred == y_test).mean()

    print(f"\n✅ Test Accuracy: {acc:.1%}")

    # Check selectivity
    confidences = y_proba.max(axis=1)
    high_conf_55 = (confidences >= 0.55).sum()
    high_conf_60 = (confidences >= 0.60).sum()
    high_conf_70 = (confidences >= 0.70).sum()

    print(f"\n📊 Confidence Distribution (Test Set):")
    print(f"   ≥55%: {high_conf_55}/{len(confidences)} ({high_conf_55/len(confidences)*100:.1f}%)")
    print(f"   ≥60%: {high_conf_60}/{len(confidences)} ({high_conf_60/len(confidences)*100:.1f}%)")
    print(f"   ≥70%: {high_conf_70}/{len(confidences)} ({high_conf_70/len(confidences)*100:.1f}%)")

    if high_conf_55 / len(confidences) > 0.80:
        print(f"\n⚠️  Model may overtrade ({high_conf_55/len(confidences)*100:.0f}% high confidence)")
    elif high_conf_55 / len(confidences) < 0.10:
        print(f"\n⚠️  Model may undertrade ({high_conf_55/len(confidences)*100:.0f}% high confidence)")
    else:
        print(f"\n✅ Good selectivity: {high_conf_55/len(confidences)*100:.1f}% high confidence trades")

    # Check class balance in predictions
    pred_up = (y_pred == 1).sum()
    pred_down = (y_pred == 0).sum()
    print(f"\n📊 Prediction Balance (Test Set):")
    print(f"   Predicted Up: {pred_up} ({pred_up/len(y_pred)*100:.1f}%)")
    print(f"   Predicted Down: {pred_down} ({pred_down/len(y_pred)*100:.1f}%)")

    print("\n" + classification_report(y_test, y_pred, target_names=['Down', 'Up'], zero_division=0))

    return model, feature_cols


def save_model(model, features, symbol, timeframe):
    """Save model."""
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
                'entry_method': 'next_bar_open_FIXED',
                'labeling_method': 'simple_future_returns',
                'training_script': 'train_model.py'
            }
        }, f)

    print(f"\n💾 Saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train model for any symbol/timeframe')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol (XAUUSD, XAGUSD, etc.)')
    parser.add_argument('--tf', type=str, required=True, help='Timeframe (5T, 15T, 30T, 1H, 4H)')
    parser.add_argument('--lookback', type=int, default=10, help='Future return lookback period (default: 10)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"TRAINING MODEL - {args.symbol} {args.tf}")
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
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    print(f"✅ Loaded {len(df):,} bars")

    # Calculate features
    print(f"\n🔧 Calculating features...")
    df = calculate_features(df)
    print(f"✅ Features calculated: {len(df):,} bars")

    # Create labels
    print(f"\n🏷️  Creating labels (simple future returns, lookback={args.lookback})...")
    df = create_simple_labels(df, lookback=args.lookback)
    print(f"✅ Labels created: {len(df):,} samples")

    # Train
    model, features = train_model(df)

    # Save
    save_model(model, features, args.symbol, args.tf)

    print(f"\n" + "="*80)
    print(f"✅ SUCCESS - {args.symbol} {args.tf} model trained!")
    print(f"\nValidate with:")
    print(f"  python validate_backtest_with_costs.py --symbol {args.symbol} --tf {args.tf} --confidence 0.70")
    print(f"="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
