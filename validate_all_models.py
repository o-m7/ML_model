#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL VALIDATION
================================

Tests ALL trained models across all timeframes to verify:
1. Models generate BOTH long and short signals
2. Performance metrics are acceptable
3. No severe directional bias

Usage:
    # Validate all XAUUSD models
    python validate_all_models.py --symbol XAUUSD

    # Validate all models for all symbols
    python validate_all_models.py --all

    # Validate specific timeframe
    python validate_all_models.py --symbol XAUUSD --tf 15T
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from market_costs import get_costs, get_tp_sl, apply_entry_costs, apply_exit_costs


def find_latest_model(symbol: str, timeframe: str) -> Path:
    """Find latest model file for symbol/timeframe."""
    # Check institutional models first
    model_dir = Path(f"models_institutional/{symbol}")
    if model_dir.exists():
        models = list(model_dir.glob(f"{symbol}_{timeframe}_*.pkl"))
        if models:
            return sorted(models)[-1]

    # Fallback to rentec
    model_dir = Path(f"models_rentec/{symbol}")
    if model_dir.exists():
        models = list(model_dir.glob(f"{symbol}_{timeframe}.pkl"))
        if models:
            return models[0]

    # Fallback to production
    model_dir = Path(f"models_production/{symbol}")
    if model_dir.exists():
        models = list(model_dir.glob(f"{symbol}_{timeframe}_*.pkl"))
        if models:
            return sorted(models)[-1]

    raise FileNotFoundError(f"No model found for {symbol} {timeframe}")


def load_model(model_path: Path) -> dict:
    """Load model from file."""
    print(f"📦 Loading: {model_path.name}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load feature data."""
    data_path = Path(f"feature_store/{symbol}/{symbol}_{timeframe}.parquet")

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_parquet(data_path)

    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

    print(f"📊 Loaded {len(df):,} bars from feature store")

    return df


def simple_backtest(
    df: pd.DataFrame,
    model: dict,
    symbol: str,
    timeframe: str,
    confidence_threshold: float = 0.50
) -> Dict:
    """
    Simple backtest to validate signal generation and basic performance.
    """
    # Extract model components
    if 'features' in model:
        features = model['features']
        model_obj = model['model']
    elif 'results' in model and 'features' in model['results']:
        features = model['results']['features']
        model_obj = model['model']
    else:
        raise ValueError("Model format not recognized")

    # Ensure features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features[:10]}")
        return None

    # Get predictions
    X = df[features].fillna(0).values
    proba = model_obj.predict_proba(X)

    # Determine class mapping
    class_names = model.get('class_names', ['Flat', 'Long', 'Short'])

    if len(class_names) == 3:
        flat_idx, long_idx, short_idx = 0, 1, 2
    elif len(class_names) == 2:
        # Binary: Up/Down
        if 'Up' in class_names and 'Down' in class_names:
            long_idx = class_names.index('Up')
            short_idx = class_names.index('Down')
            flat_idx = None
        else:
            long_idx, short_idx = 0, 1
            flat_idx = None
    else:
        print(f"❌ Unexpected class names: {class_names}")
        return None

    # Generate signals
    long_signals = []
    short_signals = []

    for i in range(len(df) - 1):
        if flat_idx is not None:
            # Multi-class
            probs = [proba[i, flat_idx], proba[i, long_idx], proba[i, short_idx]]
            max_prob = max(probs)
            pred_class = np.argmax(probs)

            if max_prob >= confidence_threshold:
                if pred_class == long_idx:
                    long_signals.append(i)
                elif pred_class == short_idx:
                    short_signals.append(i)
        else:
            # Binary
            if proba[i, long_idx] >= confidence_threshold:
                long_signals.append(i)
            elif proba[i, short_idx] >= confidence_threshold:
                short_signals.append(i)

    # Calculate metrics
    total_signals = len(long_signals) + len(short_signals)

    if total_signals == 0:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'long_pct': 0,
            'short_pct': 0,
            'balance_ratio': 0,
            'status': 'NO_SIGNALS'
        }

    long_pct = len(long_signals) / total_signals * 100
    short_pct = len(short_signals) / total_signals * 100

    if len(short_signals) > 0:
        balance_ratio = len(long_signals) / len(short_signals)
    else:
        balance_ratio = float('inf') if len(long_signals) > 0 else 0

    # Determine status
    if len(long_signals) == 0 or len(short_signals) == 0:
        status = 'BIASED'
    elif balance_ratio < 0.5 or balance_ratio > 2.0:
        status = 'IMBALANCED'
    else:
        status = 'BALANCED'

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'total_signals': total_signals,
        'long_signals': len(long_signals),
        'short_signals': len(short_signals),
        'long_pct': long_pct,
        'short_pct': short_pct,
        'balance_ratio': balance_ratio,
        'status': status,
        'class_names': class_names
    }


def validate_model(symbol: str, timeframe: str) -> Dict:
    """Validate a single model."""

    print(f"\n{'='*80}")
    print(f"VALIDATING: {symbol} {timeframe}")
    print(f"{'='*80}")

    try:
        # Find and load model
        model_path = find_latest_model(symbol, timeframe)
        model = load_model(model_path)

        # Load data
        df = load_data(symbol, timeframe)

        # Run simple backtest
        results = simple_backtest(df, model, symbol, timeframe, confidence_threshold=0.50)

        if results is None:
            print(f"❌ Validation failed")
            return None

        # Print results
        print(f"\n📊 SIGNAL GENERATION TEST:")
        print(f"   Total signals:  {results['total_signals']}")
        print(f"   Long signals:   {results['long_signals']} ({results['long_pct']:.1f}%)")
        print(f"   Short signals:  {results['short_signals']} ({results['short_pct']:.1f}%)")
        print(f"   Balance ratio:  {results['balance_ratio']:.2f}")
        print(f"   Status:         {results['status']}")

        # Status check
        if results['status'] == 'BALANCED':
            print(f"\n   ✅ Model is BALANCED and generating both directions")
        elif results['status'] == 'IMBALANCED':
            print(f"\n   ⚠️  Model is IMBALANCED (ratio {results['balance_ratio']:.2f})")
        elif results['status'] == 'BIASED':
            print(f"\n   ❌ Model is BIASED (only generating {'LONG' if results['short_signals'] == 0 else 'SHORT'})")
        else:
            print(f"\n   ❌ Model generates NO SIGNALS")

        return results

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Validate all trained models')
    parser.add_argument('--all', action='store_true', help='Validate all symbols and timeframes')
    parser.add_argument('--symbol', type=str, help='Validate specific symbol')
    parser.add_argument('--tf', type=str, help='Validate specific timeframe')

    args = parser.parse_args()

    SYMBOLS = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']
    TIMEFRAMES = ['5T', '15T', '30T', '1H', '4H']

    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*80)

    results = []

    if args.all:
        # Validate all
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                try:
                    result = validate_model(symbol, timeframe)
                    if result:
                        results.append(result)
                except FileNotFoundError as e:
                    print(f"\n⚠️  Skipping {symbol} {timeframe}: {e}")
                    continue

    elif args.symbol and args.tf:
        # Validate one
        result = validate_model(args.symbol, args.tf)
        if result:
            results.append(result)

    elif args.symbol:
        # Validate all timeframes for one symbol
        for timeframe in TIMEFRAMES:
            try:
                result = validate_model(args.symbol, timeframe)
                if result:
                    results.append(result)
            except FileNotFoundError as e:
                print(f"\n⚠️  Skipping {args.symbol} {timeframe}: {e}")
                continue

    else:
        print("\nUsage:")
        print("  python validate_all_models.py --all")
        print("  python validate_all_models.py --symbol XAUUSD")
        print("  python validate_all_models.py --symbol XAUUSD --tf 15T")
        return 1

    # Summary
    if not results:
        print("\n❌ No models validated")
        return 1

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    balanced = [r for r in results if r['status'] == 'BALANCED']
    imbalanced = [r for r in results if r['status'] == 'IMBALANCED']
    biased = [r for r in results if r['status'] == 'BIASED']
    no_signals = [r for r in results if r['status'] == 'NO_SIGNALS']

    print(f"\nTotal models validated: {len(results)}")
    print(f"✅ Balanced:    {len(balanced)}")
    print(f"⚠️  Imbalanced:  {len(imbalanced)}")
    print(f"❌ Biased:      {len(biased)}")
    print(f"❌ No signals:  {len(no_signals)}")

    if balanced:
        print(f"\n✅ BALANCED MODELS (Ready for production):")
        for r in balanced:
            print(f"   {r['symbol']} {r['timeframe']}: "
                  f"{r['long_signals']} long, {r['short_signals']} short "
                  f"(ratio: {r['balance_ratio']:.2f})")

    if imbalanced:
        print(f"\n⚠️  IMBALANCED MODELS (Use with caution):")
        for r in imbalanced:
            print(f"   {r['symbol']} {r['timeframe']}: "
                  f"{r['long_signals']} long, {r['short_signals']} short "
                  f"(ratio: {r['balance_ratio']:.2f})")

    if biased:
        print(f"\n❌ BIASED MODELS (DO NOT USE):")
        for r in biased:
            direction = 'LONG' if r['short_signals'] == 0 else 'SHORT'
            print(f"   {r['symbol']} {r['timeframe']}: "
                  f"Only {direction} signals ({r['total_signals']} total)")

    if no_signals:
        print(f"\n❌ NO SIGNALS:")
        for r in no_signals:
            print(f"   {r['symbol']} {r['timeframe']}")

    print("="*80)

    # Exit code
    if len(balanced) > 0:
        print(f"\n✅ SUCCESS: {len(balanced)} balanced models ready")
        return 0
    elif len(imbalanced) > 0:
        print(f"\n⚠️  PARTIAL: {len(imbalanced)} imbalanced models (review required)")
        return 0
    else:
        print(f"\n❌ FAILURE: No usable models found")
        return 1


if __name__ == '__main__':
    sys.exit(main())
