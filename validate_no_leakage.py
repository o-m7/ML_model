#!/usr/bin/env python3
"""
DATA LEAKAGE VALIDATOR
======================

Validates that a trained model has no data leakage issues.
Checks that test data does not overlap with training data.

Usage:
    python3 validate_no_leakage.py --model models/XAUUSD/XAUUSD_15T_*.pkl
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd


def load_model_metadata(model_path: Path) -> dict:
    """Load model and extract metadata."""
    print(f"\n{'='*80}")
    print("LOADING MODEL METADATA")
    print(f"{'='*80}")
    print(f"Model: {model_path.name}")
    
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    config = model_package['config']
    
    print(f"  Symbol: {config['symbol']}")
    print(f"  Timeframe: {config['timeframe']}")
    print(f"  Version: {model_package.get('version', '1.0.0')}")
    print(f"  Trained: {config['trained_at']}")
    
    return model_package


def validate_train_test_split(model_package: dict) -> bool:
    """Validate that train and test periods don't overlap."""
    print(f"\n{'='*80}")
    print("VALIDATING TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    
    config = model_package['config']
    
    # Check if model has train/test metadata (version 2.0.0+)
    if 'train_date_start' not in config:
        print("⚠️  WARNING: Model does not have train/test split metadata")
        print("   This is an OLD model (version 1.x) trained on ALL data")
        print("   ❌ POTENTIAL DATA LEAKAGE!")
        print("\n   Solution: Retrain model with updated jpm_production_system.py")
        return False
    
    # Extract dates
    train_start = pd.to_datetime(config['train_date_start'])
    train_end = pd.to_datetime(config['train_date_end'])
    test_start = pd.to_datetime(config['test_date_start'])
    test_end = pd.to_datetime(config['test_date_end'])
    
    print(f"\n📅 Training Period:")
    print(f"   Start: {train_start}")
    print(f"   End:   {train_end}")
    print(f"   Duration: {(train_end - train_start).days} days")
    
    print(f"\n📅 Testing Period:")
    print(f"   Start: {test_start}")
    print(f"   End:   {test_end}")
    print(f"   Duration: {(test_end - test_start).days} days")
    
    # Check for overlap
    if test_start <= train_end:
        print(f"\n❌ ERROR: Test period overlaps with training period!")
        print(f"   Train ends: {train_end}")
        print(f"   Test starts: {test_start}")
        print(f"   ❌ DATA LEAKAGE DETECTED!")
        return False
    
    # Check gap
    gap_days = (test_start - train_end).days
    print(f"\n✅ No overlap detected")
    print(f"   Gap between train and test: {gap_days} days")
    
    if gap_days == 0:
        print(f"   ✅ Perfect: Test starts immediately after training")
    elif gap_days > 7:
        print(f"   ⚠️  WARNING: Large gap ({gap_days} days) - some data not used")
    
    return True


def validate_performance_metrics(model_package: dict) -> bool:
    """Check if performance metrics are realistic."""
    print(f"\n{'='*80}")
    print("VALIDATING PERFORMANCE METRICS")
    print(f"{'='*80}")
    
    # Check if we have both train and test results
    has_train = 'backtest_train' in model_package
    has_test = 'backtest_test' in model_package
    
    if not has_train or not has_test:
        print("⚠️  WARNING: Model does not have separate train/test results")
        print("   This is an OLD model (version 1.x)")
        return False
    
    train_results = model_package['backtest_train']
    test_results = model_package['backtest_test']
    
    print(f"\n📊 Performance Comparison:")
    print(f"{'Metric':<20} {'Training':>12} {'Testing':>12} {'Difference':>12}")
    print(f"{'-'*60}")
    
    metrics_to_check = [
        ('Win Rate (%)', 'win_rate'),
        ('Profit Factor', 'profit_factor'),
        ('Sharpe Ratio', 'sharpe'),
        ('Max Drawdown (%)', 'max_drawdown'),
        ('Total Return (%)', 'total_return')
    ]
    
    issues = []
    
    for label, key in metrics_to_check:
        train_val = train_results[key]
        test_val = test_results[key]
        diff = test_val - train_val
        diff_pct = (diff / train_val * 100) if train_val != 0 else 0
        
        print(f"{label:<20} {train_val:>12.2f} {test_val:>12.2f} {diff:>12.2f}")
        
        # Check for unrealistic values
        if key == 'win_rate' and test_val > 75:
            issues.append(f"Win rate too high: {test_val:.1f}% (realistic: 50-65%)")
        
        if key == 'profit_factor' and test_val > 5:
            issues.append(f"Profit factor too high: {test_val:.2f} (realistic: 1.5-3.0)")
        
        if key == 'max_drawdown' and test_val < 2:
            issues.append(f"Max drawdown too low: {test_val:.2f}% (realistic: 3-10%)")
        
        # Check for overfitting (test much worse than train)
        if key in ['win_rate', 'profit_factor', 'sharpe'] and test_val < train_val * 0.7:
            issues.append(f"{label} drops {abs(diff_pct):.1f}% from train to test (possible overfitting)")
    
    print()
    
    if issues:
        print(f"⚠️  WARNINGS:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"✅ Performance metrics look reasonable")
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate model for data leakage')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pkl file')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1
    
    # Load model
    model_package = load_model_metadata(model_path)
    
    # Run validations
    split_valid = validate_train_test_split(model_package)
    metrics_valid = validate_performance_metrics(model_package)
    
    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")
    
    if split_valid and metrics_valid:
        print("✅ ✅ ✅  MODEL PASSED ALL VALIDATION CHECKS")
        print("\nThis model:")
        print("  ✓ Has proper train/test split")
        print("  ✓ No data leakage detected")
        print("  ✓ Realistic performance metrics")
        print("\n✅ SAFE TO USE IN PRODUCTION")
        return 0
    elif split_valid and not metrics_valid:
        print("⚠️  MODEL PASSED STRUCTURE CHECKS BUT HAS SUSPICIOUS METRICS")
        print("\nThis model:")
        print("  ✓ Has proper train/test split")
        print("  ⚠️  Performance metrics seem unrealistic")
        print("\n⚠️  USE WITH CAUTION - Review metrics carefully")
        return 0
    else:
        print("❌ ❌ ❌  MODEL FAILED VALIDATION")
        print("\nThis model:")
        print("  ❌ Has data leakage issues")
        print("  ❌ NOT SAFE FOR PRODUCTION")
        print("\n🔧 ACTION REQUIRED:")
        print("   1. Delete this model")
        print("   2. Retrain using updated jpm_production_system.py")
        print("   3. Validate new model with this script")
        return 1


if __name__ == '__main__':
    exit(main())

