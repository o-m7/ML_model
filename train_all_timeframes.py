#!/usr/bin/env python3
"""
Train all required timeframes (1T, 5T, 15T, 30T) in parallel
"""

import subprocess
import multiprocessing
from pathlib import Path
import sys

TIMEFRAMES = ["1T", "5T", "15T", "30T"]
SYMBOL = "XAUUSD"
PYTHON_BIN = sys.executable  # Use current Python executable

def train_ohlcv(timeframe):
    """Train OHLCV model for given timeframe"""
    print(f"\n{'='*60}")
    print(f"Training OHLCV {timeframe}...")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        [
            PYTHON_BIN, "citadel_v6.py",
            "--symbol", SYMBOL,
            "--timeframe", timeframe
        ],
        cwd=Path(__file__).parent
    )
    
    if result.returncode == 0:
        print(f"\n✓ OHLCV {timeframe} training complete")
    else:
        print(f"\n✗ OHLCV {timeframe} training failed")
    
    return result.returncode


def train_quote(timeframe):
    """Train Quote model for given timeframe"""
    print(f"\n{'='*60}")
    print(f"Training Quote {timeframe}...")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        [
            PYTHON_BIN, "citadel_quote_models.py",
            "--timeframe", timeframe
        ],
        cwd=Path(__file__).parent
    )
    
    if result.returncode == 0:
        print(f"\n✓ Quote {timeframe} training complete")
    else:
        print(f"\n✗ Quote {timeframe} training failed")
    
    return result.returncode


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TRAINING ALL TIMEFRAMES (1T, 5T, 15T, 30T)")
    print("="*60)
    
    # Train OHLCV models in parallel
    print("\nPhase 1: Training OHLCV Models...")
    with multiprocessing.Pool(2) as pool:  # Max 2 parallel to not overload
        ohlcv_results = pool.map(train_ohlcv, TIMEFRAMES)
    
    # Train Quote models in parallel
    print("\n\nPhase 2: Training Quote Models...")
    with multiprocessing.Pool(2) as pool:
        quote_results = pool.map(train_quote, TIMEFRAMES)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    ohlcv_success = sum(1 for r in ohlcv_results if r == 0)
    quote_success = sum(1 for r in quote_results if r == 0)
    
    print(f"\nOHLCV Models: {ohlcv_success}/{len(TIMEFRAMES)} successful")
    print(f"Quote Models: {quote_success}/{len(TIMEFRAMES)} successful")
    print(f"Total: {ohlcv_success + quote_success}/8 models trained\n")
    
    if ohlcv_success == 4 and quote_success == 4:
        print("✓ All models trained successfully!")
    else:
        print("⚠ Some models failed to train")
    
    print("="*60 + "\n")
