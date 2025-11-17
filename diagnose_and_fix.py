#!/usr/bin/env python3
"""
DIAGNOSTIC & FIX SCRIPT
=======================
Stop tweaking parameters. Find out WHY models don't learn.

This script will:
1. Load ONE segment of real data
2. Try different label definitions
3. Test feature importance
4. Remove useless features
5. Build a model that ACTUALLY learns
6. Only then integrate back to main pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt

print("=" * 80)
print("DIAGNOSTIC: WHY AREN'T MODELS LEARNING?")
print("=" * 80)

# Load real data
data_path = Path("feature_store/XAUUSD/XAUUSD_15T.parquet")
if not data_path.exists():
    # Try user's path
    data_path = Path.home() / "Desktop" / "ML_model" / "ML_model" / "feature_store" / "XAUUSD" / "XAUUSD_15T.parquet"

print(f"\n1. Loading data from: {data_path}")
df = pd.read_parquet(data_path)

if 'timestamp' not in df.columns:
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'timestamp'})

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"   Loaded: {len(df):,} rows × {len(df.columns)} columns")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Get just the first 6 months for testing
test_data = df.iloc[:30000].copy()  # ~30k bars = ~6 months of 15min data
print(f"\n2. Using first {len(test_data):,} rows for diagnostics")

# Calculate ATR if not present
if 'atr_14' not in test_data.columns:
    high_low = test_data['high'] - test_data['low']
    high_close = np.abs(test_data['high'] - test_data['close'].shift())
    low_close = np.abs(test_data['low'] - test_data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    test_data['atr_14'] = true_range.rolling(14).mean()

# Fill NaN in features first (common in technical indicators)
print("\nFilling NaN values in features...")
for col in test_data.columns:
    if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
        test_data[col] = test_data[col].ffill().bfill()

# Features
feature_cols = [col for col in test_data.columns
                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                              'bid', 'ask', 'spread', 'spread_pct', 'mid', 'atr_14']]

print(f"Features available: {len(feature_cols)}")

# TEST 1: Simple directional labels
print("\n" + "=" * 80)
print("TEST 1: Can we predict simple DIRECTION (up/down)?")
print("=" * 80)

test_data_1 = test_data.copy()
test_data_1['return_1bar'] = test_data_1['close'].pct_change(1).shift(-1)
test_data_1['direction'] = (test_data_1['return_1bar'] > 0).astype(int)

# Remove rows where we can't calculate the label
test_data_1 = test_data_1[test_data_1['return_1bar'].notna()].copy()

if len(feature_cols) == 0:
    print("\n❌ NO FEATURES FOUND! Only OHLCV data exists.")
    print("   You need to run calculate_all_features.py first!")
    exit(1)

X = test_data_1[feature_cols].values
y = test_data_1['direction'].values

print(f"\nLabel distribution:")
print(f"  UP (1):   {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"  DOWN (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

# Train/test split (time-based)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain: {len(X_train):,} samples")
print(f"Test:  {len(X_test):,} samples")

# Train simple XGBoost
print("\n" + "-" * 80)
print("Training XGBoost on directional labels...")
print("-" * 80)

model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='auc'
)

model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict_proba(X_train)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_pred)
test_auc = roc_auc_score(y_test, test_pred)

print(f"\n📊 Results:")
print(f"   Train AUC: {train_auc:.4f}")
print(f"   Test AUC:  {test_auc:.4f}")

if test_auc < 0.52:
    print("\n❌ FAILED: Model is random (AUC ≈ 0.5)")
    print("\n   DIAGNOSIS:")
    print("   1. Features don't contain predictive information")
    print("   2. OR features are all lagging indicators")
    print("   3. OR market is too efficient for 15min predictions")
    print("\n   NEXT STEPS:")
    print("   - Check feature importance")
    print("   - Try shorter prediction horizons")
    print("   - Add microstructure features (order flow, imbalance)")
else:
    print(f"\n✅ Model is learning! (AUC = {test_auc:.4f})")

# Feature importance
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Top 20)")
print("=" * 80)

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(20).to_string(index=False))

# Save top features
top_features = importance.head(20)['feature'].tolist()
print(f"\n💾 Top 20 features saved to top_features.txt")
with open('top_features.txt', 'w') as f:
    f.write('\n'.join(top_features))

# TEST 2: Try TP/SL labels
print("\n" + "=" * 80)
print("TEST 2: Can we predict TP/SL outcomes?")
print("=" * 80)

def create_tpsl_labels(df, tp_mult=2.0, sl_mult=1.0, max_bars=8):
    """Create TP/SL labels"""
    labels = []

    for i in range(len(df) - max_bars):
        entry = df['close'].iloc[i]
        atr = df['atr_14'].iloc[i]

        tp = entry + (tp_mult * atr)
        sl = entry - (sl_mult * atr)

        hit_tp = False
        for j in range(1, max_bars + 1):
            if i + j >= len(df):
                break
            high = df['high'].iloc[i + j]
            low = df['low'].iloc[i + j]

            if high >= tp:
                hit_tp = True
                break
            if low <= sl:
                break

        labels.append(1 if hit_tp else 0)

    # Pad with NaN
    labels.extend([0] * max_bars)
    return labels

test_data['tpsl_label'] = create_tpsl_labels(test_data)
# Features already filled from TEST 1, just ensure no remaining NaN
test_data = test_data.fillna(0)

X2 = test_data[feature_cols].values
y2 = test_data['tpsl_label'].values

print(f"\nTP/SL Label distribution:")
print(f"  TP HIT (1): {(y2==1).sum()} ({(y2==1).mean()*100:.1f}%)")
print(f"  SL HIT (0): {(y2==0).sum()} ({(y2==0).mean()*100:.1f}%)")

split_idx2 = int(len(X2) * 0.8)
X_train2, X_test2 = X2[:split_idx2], X2[split_idx2:]
y_train2, y_test2 = y2[:split_idx2], y2[split_idx2:]

model2 = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='auc'
)

model2.fit(X_train2, y_train2)

train_pred2 = model2.predict_proba(X_train2)[:, 1]
test_pred2 = model2.predict_proba(X_test2)[:, 1]

train_auc2 = roc_auc_score(y_train2, train_pred2)
test_auc2 = roc_auc_score(y_test2, test_pred2)

print(f"\n📊 TP/SL Results:")
print(f"   Train AUC: {train_auc2:.4f}")
print(f"   Test AUC:  {test_auc2:.4f}")

if test_auc2 < 0.52:
    print("\n❌ TP/SL labels also random!")
else:
    print(f"\n✅ TP/SL labels work! (AUC = {test_auc2:.4f})")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\n1. Directional prediction (1-bar ahead):")
print(f"   AUC: {test_auc:.4f} - {'✅ WORKS' if test_auc >= 0.52 else '❌ RANDOM'}")

print(f"\n2. TP/SL prediction ({2.0}R TP, {1.0}R SL, {8} bars):")
print(f"   AUC: {test_auc2:.4f} - {'✅ WORKS' if test_auc2 >= 0.52 else '❌ RANDOM'}")

print(f"\n3. Feature importance:")
print(f"   Top 5 features: {', '.join(top_features[:5])}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if test_auc < 0.52 and test_auc2 < 0.52:
    print("\n❌ CRITICAL: Neither label type works!")
    print("\n   ROOT CAUSES:")
    print("   1. Technical indicators are lagging (already priced in)")
    print("   2. 15-minute timeframe too efficient")
    print("   3. Need microstructure/order flow features")
    print("\n   FIXES:")
    print("   - Add volume profile features")
    print("   - Add quote imbalance (if you have L2 data)")
    print("   - Try 5-minute timeframe")
    print("   - Consider regime-based models")

elif test_auc >= 0.52:
    print(f"\n✅ Directional labels WORK (AUC {test_auc:.4f})")
    print(f"\n   Use these top {len(top_features)} features")
    print(f"   Predicted direction → trade that direction")
    print(f"   This is your viable strategy!")

elif test_auc2 >= 0.52:
    print(f"\n✅ TP/SL labels WORK (AUC {test_auc2:.4f})")
    print(f"\n   Continue with TP/SL approach")
    print(f"   Use these top {len(top_features)} features")

print("\n" + "=" * 80)
