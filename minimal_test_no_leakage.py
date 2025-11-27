#!/usr/bin/env python3
"""
Minimal test with ONLY past bar features to verify no leakage.
If this shows PF > 3 or WR > 65%, the labeling itself is broken.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import lightgbm as lgb

print("="*60)
print("MINIMAL NO-LEAKAGE TEST")
print("="*60)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

base_path = Path(config['paths']['base'])
symbol = config['data']['symbol']
timeframe = config['data']['primary_timeframe']

print(f"\nLoading {symbol} {timeframe} data...")
df = pd.read_parquet(base_path / f"{symbol}/{symbol}_{timeframe}.parquet")

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

df = df.sort_index()
print(f"Loaded: {len(df)} rows")

# Use last 6 months (more data for better learning)
df = df.tail(6 * 30 * 24 * (60 // int(timeframe.replace('T', ''))))
print(f"Using last 6 months: {len(df)} rows\n")

# CRITICAL: Create labels FIRST
print("Creating labels...")
from labeling import TripleBarrierLabeler

labeler = TripleBarrierLabeler(
    take_profit_atr=2.0,
    stop_loss_atr=1.0,
    time_barrier_bars=12,
    min_return_threshold=0.0001
)

df = labeler.label_data(df)

# ONLY use features from PAST bars (shift by 1 bar - standard lag)
print("\nCreating SAFE features (only past data)...")
safe_features = pd.DataFrame(index=df.index)

# Price features - lagged by 1 bar (standard)
safe_features['close_lag1'] = df['close'].shift(1)
safe_features['high_lag1'] = df['high'].shift(1)
safe_features['low_lag1'] = df['low'].shift(1)
safe_features['volume_lag1'] = df['volume'].shift(1)

# ATR from 1 bar ago
safe_features['ATR_lag1'] = df['ATR'].shift(1)

# Simple moving averages (using lagged data)
safe_features['sma_10'] = df['close'].shift(1).rolling(10).mean()
safe_features['sma_20'] = df['close'].shift(1).rolling(20).mean()
safe_features['sma_50'] = df['close'].shift(1).rolling(50).mean()

# Price changes (using lagged data) - use safe names
safe_features['ret_1_lag1'] = df['close'].shift(1).pct_change(1)
safe_features['ret_5_lag1'] = df['close'].shift(1).pct_change(5)

# Volatility (using lagged data)
safe_features['volatility_10'] = df['close'].shift(1).pct_change().rolling(10).std()

# RSI (simple version, using lagged data)
delta = df['close'].shift(1).diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
safe_features['rsi_14'] = 100 - (100 / (1 + gain / loss.replace(0, 0.0001)))

# That's 12 simple features, all lagged by 1 bar

# Drop any existing feature columns to avoid conflicts
existing_feature_cols = [c for c in df.columns if c not in ['label', 'return', 'bars_held', 'barrier_hit', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'ATR']]
if existing_feature_cols:
    print(f"Dropping {len(existing_feature_cols)} existing feature columns")
    df = df.drop(columns=existing_feature_cols)

# Merge with labels
df = df.join(safe_features)
df = df.dropna()

print(f"Features: {list(safe_features.columns)}")
print(f"All features lagged by 2+ bars")
print(f"After dropna: {len(df)} rows\n")

# Prepare data
df_binary = df[df['label'] != 0].copy()
df_binary['label'] = (df_binary['label'] == 1).astype(int)

feature_cols = safe_features.columns.tolist()
X = df_binary[feature_cols].values
y = df_binary['label'].values
returns = df_binary['return'].values

# Split
n = len(X)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
returns_test = returns[val_end:]

print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}\n")

# Train
print("Training LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

model = lgb.train(
    params, train_data,
    num_boost_round=200,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

# Evaluate
y_pred_proba = model.predict(X_test)

# Show prediction distribution
print(f"\nPrediction Statistics:")
print(f"  Min:  {y_pred_proba.min():.4f}")
print(f"  Max:  {y_pred_proba.max():.4f}")
print(f"  Mean: {y_pred_proba.mean():.4f}")
print(f"  Std:  {y_pred_proba.std():.4f}")
print(f"  Predictions > 0.50: {(y_pred_proba > 0.50).sum()}/{len(y_pred_proba)}")
print(f"  Predictions > 0.45: {(y_pred_proba > 0.45).sum()}/{len(y_pred_proba)}")

# Use lower threshold to get some trades
threshold = 0.45
y_pred = (y_pred_proba > threshold).astype(int)

# Calculate metrics
trades = returns_test[y_pred == 1]

if len(trades) > 0:
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    
    win_rate = len(wins) / len(trades)
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
else:
    win_rate = 0
    pf = 0

print("\n" + "="*60)
print("MINIMAL TEST RESULTS")
print("="*60)
print(f"Confidence Threshold: {threshold}")
print(f"Profit Factor: {pf:.2f}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Total Trades: {len(trades)}")
print("="*60)

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if len(trades) == 0:
    print("⚠️  NO TRADES - Model has no confidence")
    print("Possible reasons:")
    print("  1. Features don't capture predictive patterns")
    print("  2. Training data too small (3 months)")
    print("  3. Labeling parameters too strict")
    print("\nSuggestions:")
    print("  - Use 6 months of data instead of 3")
    print("  - Lower TP target (1.5x ATR instead of 2.0x)")
    print("  - Add more features (but keep lagged)")
elif pf > 3.0 or win_rate > 0.65:
    print("🚨 STILL HAS LEAKAGE!")
    print("Even with ultra-safe 1-bar lagged features, performance is unrealistic.")
    print("\nThis means:")
    print("  - Triple-barrier labeling is using current bar data")
    print("  - OR data file contains pre-calculated features that leak")
    print("\nAction required:")
    print("  1. Audit triple-barrier calculation")
    print("  2. Check if input data has future-looking features")
    print("  3. Verify ATR calculation doesn't include current bar")
elif pf > 1.3 and win_rate > 0.50:
    print("✅ NO LEAKAGE - Performance is REALISTIC!")
    print("This is what proper ML trading looks like.")
    print("\nYour previous results (PF=7.9, WR=85%) were from:")
    print("  - Feature leakage (4H features, returns, momentum)")
    print("  - NOT from labeling")
    print("\nRecommendation:")
    print("  - Build features using ONLY this pattern:")
    print("    * df['feature'] = df['source'].shift(1).transform(...)")
    print("  - Never use higher timeframe features")
    print("  - All features must be lagged by at least 1 bar")
else:
    print("⚠️  Performance suboptimal but NOT leaking")
    print(f"PF={pf:.2f} and WR={win_rate:.1%} are low but realistic.")
    print("\nThis could mean:")
    print("  1. Not enough features (only 12 basic ones)")
    print("  2. Need better feature engineering")
    print("  3. 15T timeframe too noisy for simple features")
    print("\nSuggestions:")
    print("  - Try 30T or 1H (less noise)")
    print("  - Add more lagged technical indicators")
    print("  - Use longer training window (6 months)")

print("="*60 + "\n")