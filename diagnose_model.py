#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT - Find out why model predicts on every bar
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shared_features import calculate_features


def diagnose_model(symbol='XAUUSD', timeframe='15T'):
    """Complete diagnostic of model behavior."""

    print("\n" + "="*80)
    print("MODEL DIAGNOSTIC - COMPLETE AUDIT")
    print("="*80 + "\n")

    # 1. Find and load model
    print("1️⃣ MODEL LOADING")
    print("-" * 80)

    model_paths = [
        Path(f"models_rentec/{symbol}/{symbol}_{timeframe}.pkl"),
        Path(f"models_production/{symbol}/{symbol}_{timeframe}_PRODUCTION_READY.pkl"),
        Path(f"models_fast/{symbol}/{symbol}_{timeframe}_READY_20251104_205719.pkl"),
    ]

    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break

    if not model_path:
        print("❌ NO MODEL FOUND")
        return

    print(f"✅ Found model: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")
    print(f"   Modified: {pd.Timestamp.fromtimestamp(model_path.stat().st_mtime)}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"\n   Model keys: {list(model_data.keys())}")

    if 'metadata' in model_data:
        print(f"   Metadata: {model_data['metadata']}")

    # Get model object
    if 'model' in model_data:
        model = model_data['model']
    else:
        print("❌ No 'model' key in pickle file")
        return

    if 'features' in model_data:
        features = model_data['features']
    elif 'results' in model_data and 'features' in model_data['results']:
        features = model_data['results']['features']
    else:
        print("❌ No features found in model")
        return

    print(f"   Features: {len(features)} total")
    print(f"   First 5: {features[:5]}")

    # 2. Load data
    print("\n2️⃣ DATA LOADING")
    print("-" * 80)

    data_path = Path(f"feature_store/{symbol}/{symbol}_{timeframe}.parquet")
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df):,} bars")

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Calculate features
    print(f"   Calculating features...")
    df = calculate_features(df)
    print(f"   After features: {len(df):,} bars")

    # Check if we have all required features
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing[:10]}")
        return

    print(f"✅ All features present")

    # 3. Get predictions
    print("\n3️⃣ MODEL PREDICTIONS")
    print("-" * 80)

    X = df[features].fillna(0).values

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print(f"✅ Predictions shape: {predictions.shape}")
    print(f"   Probabilities shape: {probabilities.shape}")
    print(f"   Num classes: {probabilities.shape[1]}")

    # Analyze predictions
    unique_preds, counts_preds = np.unique(predictions, return_counts=True)
    print(f"\n   Prediction distribution:")
    for pred, count in zip(unique_preds, counts_preds):
        pct = count / len(predictions) * 100
        print(f"      Class {pred}: {count:,} ({pct:.1f}%)")

    # Analyze confidence
    confidences = probabilities.max(axis=1)

    print(f"\n   Confidence distribution:")
    print(f"      Mean: {confidences.mean():.3f}")
    print(f"      Median: {np.median(confidences):.3f}")
    print(f"      Min: {confidences.min():.3f}")
    print(f"      Max: {confidences.max():.3f}")

    # Confidence buckets
    for threshold in [0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        count = (confidences >= threshold).sum()
        pct = count / len(confidences) * 100
        print(f"      ≥{threshold:.0%}: {count:,} bars ({pct:.1f}%)")

    # 4. Simulate backtest entry logic
    print("\n4️⃣ BACKTEST SIMULATION (Confidence ≥ 55%)")
    print("-" * 80)

    confidence_threshold = 0.55
    trades_that_would_trigger = (confidences >= confidence_threshold).sum()

    print(f"   Bars with confidence ≥ {confidence_threshold}: {trades_that_would_trigger:,} / {len(df):,}")
    print(f"   That's {trades_that_would_trigger / len(df) * 100:.1f}% of all bars!")

    if trades_that_would_trigger > len(df) * 0.5:
        print(f"\n   🚨 PROBLEM DETECTED:")
        print(f"      Model predicts with high confidence on >50% of bars")
        print(f"      This is NOT pattern recognition - it's random!")
        print(f"\n   Possible causes:")
        print(f"      1. Model was trained on garbage data")
        print(f"      2. Model has severe overfitting")
        print(f"      3. Features are leaking future information")
        print(f"      4. Model is predicting randomly with high confidence")

    # 5. Sample predictions
    print("\n5️⃣ SAMPLE PREDICTIONS (Random 10 bars)")
    print("-" * 80)

    sample_indices = np.random.choice(len(df), min(10, len(df)), replace=False)

    print(f"\n   {'Index':<8} {'Pred':<6} {'Conf':<8} {'Class 0':<10} {'Class 1':<10}")
    print(f"   {'-'*60}")

    for idx in sorted(sample_indices):
        pred = predictions[idx]
        conf = confidences[idx]
        prob_0 = probabilities[idx][0]
        prob_1 = probabilities[idx][1]
        print(f"   {idx:<8} {pred:<6} {conf:>6.1%}   {prob_0:>8.1%}   {prob_1:>8.1%}")

    # 6. Decision
    print("\n6️⃣ DIAGNOSIS")
    print("-" * 80)

    if trades_that_would_trigger < len(df) * 0.05:
        print("✅ Model is VERY selective (trades <5% of bars) - Probably too strict")
    elif trades_that_would_trigger < len(df) * 0.20:
        print("✅ Model is selective (trades <20% of bars) - Good pattern recognition")
    elif trades_that_would_trigger < len(df) * 0.50:
        print("⚠️  Model is moderately selective (20-50% of bars) - May be overfitting")
    else:
        print("❌ MODEL IS BROKEN - Predicts on >50% of bars")
        print("   This model cannot recognize patterns and should NOT be used!")
        print("\n   ROOT CAUSE:")

        # Check if it's the demo model
        if 'metadata' in model_data and model_data['metadata'].get('data_type') == 'synthetic_demo':
            print("   ⚠️  This is the DEMO model trained on synthetic data!")
            print("   ⚠️  It will fail on real market data!")
        else:
            print("   ⚠️  Model was likely trained with bugs (look-ahead bias, wrong labels, etc)")

        print("\n   SOLUTION:")
        print("   1. Delete this model")
        print("   2. Train fresh model with REAL data and FIXED training system")
        print("   3. Expected result: ~10-20% of bars trigger trades")

    print("\n" + "="*80)
    print("END DIAGNOSTIC")
    print("="*80 + "\n")


if __name__ == '__main__':
    diagnose_model()
