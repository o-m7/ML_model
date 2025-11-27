"""
Quick test training - runs a minimal version to verify everything works.
Uses small data subset and single model for fast validation.
"""

import yaml
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

print("\n" + "="*60)
print("QUICK TEST TRAINING")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load config
print("Loading configuration...")
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded: {config['data']['symbol']} {config['data']['primary_timeframe']}\n")
except Exception as e:
    print(f"❌ Failed to load config: {e}")
    sys.exit(1)

# Import modules
print("Importing pipeline modules...")
try:
    from data_loader import DataLoader
    from labeling import TripleBarrierLabeler
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer
    print("✅ All modules imported\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Step 1: Load data (use subset for speed)
print("="*60)
print("STEP 1: LOADING DATA (subset for speed)")
print("="*60)

data_loader = DataLoader(config)
df = data_loader.load_data()

# Use only last 3 months for quick test
df_subset = df.tail(3 * 30 * 24)  # Last 3 months
print(f"Using subset: {len(df_subset)} rows (last 3 months)")
print(f"Date range: {df_subset.index.min()} to {df_subset.index.max()}\n")

# Step 2: Labeling
print("="*60)
print("STEP 2: TRIPLE-BARRIER LABELING")
print("="*60)

labeler = TripleBarrierLabeler(
    take_profit_atr=2.0,
    stop_loss_atr=1.0,
    time_barrier_bars=24,
    min_return_threshold=0.0001
)

df_subset = labeler.label_data(df_subset)
print("")

# Step 3: Feature Engineering
print("="*60)
print("STEP 3: FEATURE ENGINEERING")
print("="*60)

feature_engineer = FeatureEngineer(config)
df_subset = feature_engineer.engineer_features(df_subset)
print("")

# Step 4: Train/Val/Test Split
print("="*60)
print("STEP 4: DATA SPLITTING")
print("="*60)

# Simple 60/20/20 split for quick test
n = len(df_subset)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train_df = df_subset.iloc[:train_end]
val_df = df_subset.iloc[train_end:val_end]
test_df = df_subset.iloc[val_end:]

print(f"Train: {len(train_df)} rows")
print(f"Val:   {len(val_df)} rows")
print(f"Test:  {len(test_df)} rows\n")

# Step 5: Train single model (LightGBM - fastest)
print("="*60)
print("STEP 5: TRAINING MODEL (LightGBM only)")
print("="*60)

trainer = ModelTrainer(config)

# Prepare data
X_train, y_train, feature_names = trainer.prepare_data(train_df)
X_val, y_val, _ = trainer.prepare_data(val_df)
X_test, y_test, _ = trainer.prepare_data(test_df)

print(f"Features: {len(feature_names)}")
print(f"Train samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")
print(f"Class balance: {np.bincount(y_train)}\n")

# Train
print("Training LightGBM...")
model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
print("✅ Training complete\n")

# Step 6: Quick evaluation
print("="*60)
print("STEP 6: EVALUATION")
print("="*60)

# Predict on test set
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate metrics
from validator import PerformanceCalculator
perf_calc = PerformanceCalculator()

# Get returns from test set
returns = test_df[test_df['label'] != 0]['return'].values

metrics = perf_calc.calculate_metrics(y_test, y_pred, returns)

print("Test Set Performance:")
print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
print(f"  Win Rate: {metrics['win_rate']:.2%}")
print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"  Sharpe: {metrics['sharpe']:.2f}")
print(f"  R-multiple: {metrics['r_multiple']:.2f}")
print(f"  Total Trades: {metrics['total_trades']}")
print(f"  Expected Value: ${metrics['expected_value_dollar']:.4f}")

# Summary
print("\n" + "="*60)
print("QUICK TEST COMPLETE")
print("="*60)

if metrics['profit_factor'] >= 1.5 and metrics['win_rate'] >= 0.50:
    print("✅ Results look reasonable - ready for full training!")
    print("\nNext step: Run full pipeline:")
    print("  python train_xauusd_models.py")
elif metrics['profit_factor'] > 5.0 or metrics['win_rate'] > 0.80:
    print("⚠️  WARNING: Results too good - possible data leakage!")
    print("Check feature engineering for lookahead bias")
else:
    print("⚠️  Results below targets - may need tuning")
    print("This is just a quick test - full pipeline may perform better")

print("\nFinished: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("="*60 + "\n")