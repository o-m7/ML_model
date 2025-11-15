#!/usr/bin/env python3
"""
RETRAIN ALL MODELS - Temporal Training (Proper Time-Based Split)
================================================================
Retrains ALL models using CHRONOLOGICAL splits (no data leakage).

This fixes the 0% backtest win rate issue caused by random shuffling.

Usage:
    python retrain_all_temporal.py
    python retrain_all_temporal.py --symbols XAUUSD XAGUSD
    python retrain_all_temporal.py --timeframes 15T 30T
"""

import argparse
import subprocess
import sys
from pathlib import Path


# All active models
ALL_MODELS = [
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
    """Check if data exists for symbol/timeframe."""
    data_path = Path(f"feature_store/{symbol}/{symbol}_{timeframe}.parquet")
    return data_path.exists()


def retrain_model(symbol, timeframe, lookback=10, test_size=0.2):
    """Retrain a single model using temporal split."""
    print(f"\n{'='*80}")
    print(f"RETRAINING: {symbol} {timeframe}")
    print(f"{'='*80}")

    # Check data exists
    if not check_data_exists(symbol, timeframe):
        print(f"⚠️  SKIP: No data found for {symbol} {timeframe}")
        return False

    # Run training script
    cmd = [
        'python', 'train_model_temporal.py',
        '--symbol', symbol,
        '--tf', timeframe,
        '--lookback', str(lookback),
        '--test-size', str(test_size)
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n✅ SUCCESS: {symbol} {timeframe}")
        return True
    else:
        print(f"\n❌ FAILED: {symbol} {timeframe}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Retrain all models with TEMPORAL splits')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to retrain (default: all)')
    parser.add_argument('--timeframes', nargs='+', default=None,
                       help='Timeframes to retrain (default: all)')
    parser.add_argument('--lookback', type=int, default=10,
                       help='Future return lookback (default: 10)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2 = 20%%)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without doing it')

    args = parser.parse_args()

    # Filter models
    models_to_train = []
    for symbol, timeframe in ALL_MODELS:
        if args.symbols and symbol not in args.symbols:
            continue
        if args.timeframes and timeframe not in args.timeframes:
            continue
        models_to_train.append((symbol, timeframe))

    if not models_to_train:
        print("❌ No models match the filters")
        return 1

    print("\n" + "="*80)
    print("TEMPORAL MODEL RETRAINING")
    print("="*80)
    print(f"\nModels to retrain: {len(models_to_train)}")
    for symbol, timeframe in models_to_train:
        status = "✅" if check_data_exists(symbol, timeframe) else "❌"
        print(f"  {status} {symbol} {timeframe}")

    if args.dry_run:
        print("\n[DRY RUN] No models trained")
        return 0

    input(f"\nPress ENTER to start retraining {len(models_to_train)} models (Ctrl+C to cancel)...")

    # Train all models
    results = []
    for symbol, timeframe in models_to_train:
        success = retrain_model(symbol, timeframe, args.lookback, args.test_size)
        results.append((symbol, timeframe, success))

    # Summary
    print("\n" + "="*80)
    print("RETRAINING SUMMARY")
    print("="*80)

    successes = [r for r in results if r[2]]
    failures = [r for r in results if not r[2]]

    print(f"\n✅ Successful: {len(successes)}/{len(results)}")
    for symbol, timeframe, _ in successes:
        print(f"   ✅ {symbol} {timeframe}")

    if failures:
        print(f"\n❌ Failed: {len(failures)}/{len(results)}")
        for symbol, timeframe, _ in failures:
            print(f"   ❌ {symbol} {timeframe}")

    print(f"\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Backtest the retrained models:")
    print("   python run_model_backtest.py --symbol XAUUSD --timeframe 15T")
    print("\n2. Compare with old models to verify improvement")
    print("\n3. Deploy only models that:")
    print("   - Test accuracy ≥ 52% (better than random)")
    print("   - Backtest win rate ≥ 50%")
    print("   - Pass prop-firm challenge")
    print()

    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
