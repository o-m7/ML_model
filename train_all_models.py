#!/usr/bin/env python3
"""
TRAIN ALL MODELS - Batch training for multiple symbols/timeframes
==================================================================

Trains models for:
- XAUUSD: 5T, 15T, 30T, 1H
- XAGUSD: 5T, 15T, 30T, 1H

Usage:
    python train_all_models.py
"""

import subprocess
import sys
from pathlib import Path


# Models to train
MODELS = [
    ('XAUUSD', '5T'),
    ('XAUUSD', '15T'),
    ('XAUUSD', '30T'),
    ('XAUUSD', '1H'),
    ('XAGUSD', '5T'),
    ('XAGUSD', '15T'),
    ('XAGUSD', '30T'),
    ('XAGUSD', '1H'),
]


def check_data_exists(symbol, timeframe):
    """Check if data file exists."""
    data_path = Path(f"feature_store/{symbol}/{symbol}_{timeframe}.parquet")
    return data_path.exists()


def train_model(symbol, timeframe):
    """Train a single model."""
    print("\n" + "="*80)
    print(f"TRAINING: {symbol} {timeframe}")
    print("="*80)

    # Check if data exists
    if not check_data_exists(symbol, timeframe):
        print(f"⚠️  SKIPPING - Data not found: feature_store/{symbol}/{symbol}_{timeframe}.parquet")
        return False

    # Train model
    cmd = ['python', 'train_model.py', '--symbol', symbol, '--tf', timeframe]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"✅ SUCCESS - {symbol} {timeframe}")
        return True
    else:
        print(f"❌ FAILED - {symbol} {timeframe}")
        return False


def main():
    print("\n" + "="*80)
    print("BATCH TRAINING - ALL MODELS")
    print("="*80)

    # Check which data files exist
    print("\n📊 Checking available data...")
    available = []
    missing = []

    for symbol, timeframe in MODELS:
        if check_data_exists(symbol, timeframe):
            available.append((symbol, timeframe))
            print(f"  ✅ {symbol} {timeframe}")
        else:
            missing.append((symbol, timeframe))
            print(f"  ❌ {symbol} {timeframe}")

    if not available:
        print("\n❌ No data files found! Please download data first.")
        return 1

    if missing:
        print(f"\n⚠️  {len(missing)} models will be skipped (no data)")

    print(f"\n🔥 Training {len(available)} models...\n")

    # Train all available models
    success = []
    failed = []

    for symbol, timeframe in available:
        if train_model(symbol, timeframe):
            success.append((symbol, timeframe))
        else:
            failed.append((symbol, timeframe))

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    print(f"\n✅ Successful: {len(success)}/{len(available)}")
    for symbol, timeframe in success:
        print(f"  ✅ {symbol} {timeframe}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(available)}")
        for symbol, timeframe in failed:
            print(f"  ❌ {symbol} {timeframe}")

    if missing:
        print(f"\n⚠️  Skipped: {len(missing)} (no data)")
        for symbol, timeframe in missing:
            print(f"  ⚠️  {symbol} {timeframe}")

    print("\n" + "="*80)

    if len(success) == len(available):
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    else:
        print(f"⚠️  {len(failed)} models failed to train")

    print("="*80 + "\n")

    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
