#!/usr/bin/env python3
"""
FULL SYSTEM AUDIT - Find ALL bugs causing bad performance
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shared_features import calculate_features


print("\n" + "="*80)
print("FULL SYSTEM AUDIT")
print("="*80 + "\n")

# 1. Load model
print("1️⃣ LOADING MODEL")
print("-" * 80)

model_path = Path("models_rentec/XAUUSD/XAUUSD_15T.pkl")
if not model_path.exists():
    model_path = Path("models_production/XAUUSD/XAUUSD_15T_PRODUCTION_READY.pkl")

if not model_path.exists():
    print("❌ No model found")
    sys.exit(1)

print(f"✅ Loading: {model_path}")

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

print(f"   Keys: {list(model_data.keys())}")

if 'metadata' in model_data:
    print(f"   Metadata: {model_data['metadata']}")

model = model_data['model']
features = model_data.get('features') or model_data['results']['features']

print(f"   Features: {len(features)}")

# 2. Load data
print("\n2️⃣ LOADING DATA")
print("-" * 80)

df = pd.read_parquet("feature_store/XAUUSD/XAUUSD_15T.parquet")
print(f"✅ Loaded {len(df):,} bars")

# Keep only OHLCV
df = df[['open', 'high', 'low', 'close', 'volume']].copy()

# Calculate features
df = calculate_features(df)
print(f"✅ After features: {len(df):,} bars")

# 3. Get predictions
print("\n3️⃣ MODEL PREDICTIONS")
print("-" * 80)

X = df[features].fillna(0).values
predictions = model.predict(X)
probabilities = model.predict_proba(X)
confidences = probabilities.max(axis=1)

print(f"Predictions: {predictions.shape}")
print(f"Probabilities: {probabilities.shape}")

# Analyze predictions
unique_preds, counts = np.unique(predictions, return_counts=True)
print(f"\nPrediction distribution:")
for pred, count in zip(unique_preds, counts):
    print(f"   Class {pred}: {count:,} ({count/len(predictions)*100:.1f}%)")

# Analyze confidence
print(f"\nConfidence stats:")
print(f"   Mean: {confidences.mean():.3f}")
print(f"   Median: {np.median(confidences):.3f}")
print(f"   Std: {confidences.std():.3f}")

print(f"\nConfidence distribution:")
for threshold in [0.50, 0.55, 0.60, 0.70, 0.80]:
    count = (confidences >= threshold).sum()
    pct = count / len(confidences) * 100
    print(f"   ≥{threshold:.0%}: {count:,} ({pct:.1f}%)")

# 4. Problem diagnosis
print("\n4️⃣ PROBLEM DIAGNOSIS")
print("-" * 80)

trades_at_55 = (confidences >= 0.55).sum()
pct_at_55 = trades_at_55 / len(df) * 100

print(f"Bars with confidence ≥55%: {trades_at_55:,} / {len(df):,} ({pct_at_55:.1f}%)")

if pct_at_55 > 80:
    print(f"\n❌ CRITICAL: Model predicts on {pct_at_55:.0f}% of bars!")
    print(f"   This is NOT pattern recognition.")
    print(f"\n   ROOT CAUSE CHECK:")

    # Check if probabilities are reasonable
    avg_prob_class0 = probabilities[:, 0].mean()
    avg_prob_class1 = probabilities[:, 1].mean()

    print(f"   Average P(Class 0): {avg_prob_class0:.3f}")
    print(f"   Average P(Class 1): {avg_prob_class1:.3f}")

    if abs(avg_prob_class0 - 0.5) < 0.05:
        print(f"\n   ⚠️  Model probabilities are ~50/50 (random guessing)")
        print(f"       But confidence is high because it always picks the slightly higher one")
        print(f"\n   CAUSE: Model was trained on bad data or has severe issues")

    # Sample predictions
    print(f"\n   Sample predictions (first 20 bars):")
    print(f"   {'Bar':<6} {'Pred':<6} {'Conf':<8} {'P(0)':<8} {'P(1)':<8}")
    for i in range(min(20, len(df))):
        print(f"   {i:<6} {predictions[i]:<6} {confidences[i]:>6.1%} {probabilities[i,0]:>6.1%} {probabilities[i,1]:>6.1%}")

    print(f"\n   SOLUTION:")
    print(f"   The model is fundamentally broken. It needs to be retrained from scratch.")
    print(f"   Expected: <20% of bars should trigger trades")
    print(f"   Actual: {pct_at_55:.0f}% of bars trigger trades")

elif pct_at_55 > 50:
    print(f"\n⚠️  WARNING: Model predicts on {pct_at_55:.0f}% of bars")
    print(f"   This is too high. Model may be overfitting or have training issues.")

elif pct_at_55 > 20:
    print(f"\n⚠️  Model predicts on {pct_at_55:.0f}% of bars")
    print(f"   This is moderately selective but could be better.")

else:
    print(f"\n✅ Model is selective, predicting on only {pct_at_55:.1f}% of bars")
    print(f"   This indicates good pattern recognition.")

# 5. Check for numerical issues
print("\n5️⃣ DATA QUALITY CHECK")
print("-" * 80)

print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
print(f"ATR range: {df['atr14'].min():.4f} - {df['atr14'].max():.4f}")
print(f"Any NaN in features: {df[features].isna().any().any()}")
print(f"Any inf in features: {np.isinf(df[features].values).any()}")

print("\n" + "="*80)
print("END AUDIT")
print("="*80 + "\n")
