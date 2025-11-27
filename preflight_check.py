"""
Pre-flight check before training - ensures everything is ready.

Usage:
    python preflight_check.py --timeframe 1H
    python preflight_check.py --all  # Check all timeframes
"""

import pandas as pd
from pathlib import Path
import argparse

def check_unified_file(symbol: str, timeframe: str, base_path: str):
    """Check single unified file."""
    unified_dir = Path(base_path) / "unified" / symbol
    file_path = unified_dir / f"{symbol}_{timeframe}_unified.parquet"
    
    print(f"\n{'='*70}")
    print(f"Checking: {symbol} {timeframe}")
    print(f"{'='*70}")
    
    # Check existence
    if not file_path.exists():
        print(f"❌ FAILED: File not found")
        print(f"   Expected: {file_path}")
        print(f"   Run: python rebuild_unified_data.py --symbols {symbol} --timeframes {timeframe}")
        return False
    
    print(f"✅ File exists: {file_path.name}")
    print(f"   Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load and validate
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ File loads successfully")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ FAILED: Cannot load file")
        print(f"   Error: {e}")
        return False
    
    # Check required columns
    required = ['close', 'high', 'low', 'open', 'ATR']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"❌ FAILED: Missing required columns: {missing}")
        return False
    
    print(f"✅ All required columns present")
    
    # Check data quality
    nan_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
    
    if nan_pct > 20:
        print(f"⚠️  WARNING: High NaN percentage: {nan_pct:.2f}%")
        print(f"   Training may be affected")
    else:
        print(f"✅ NaN percentage acceptable: {nan_pct:.2f}%")
    
    # Check time range
    duration_days = (df.index.max() - df.index.min()).days
    
    if duration_days < 180:
        print(f"⚠️  WARNING: Short history: {duration_days} days")
        print(f"   Minimum 180 days recommended for 6-month training windows")
    else:
        print(f"✅ Sufficient history: {duration_days} days")
    
    # Check data integrity
    if not df.index.is_monotonic_increasing:
        print(f"❌ FAILED: Non-monotonic timestamps")
        return False
    
    print(f"✅ Monotonic timestamps")
    
    if df.index.duplicated().sum() > 0:
        print(f"❌ FAILED: Duplicate timestamps: {df.index.duplicated().sum()}")
        return False
    
    print(f"✅ No duplicate timestamps")
    
    # Check ATR validity
    atr_valid = (df['ATR'] > 0).sum()
    atr_invalid = len(df) - atr_valid
    
    if atr_invalid > len(df) * 0.05:
        print(f"⚠️  WARNING: {atr_invalid} invalid ATR values ({atr_invalid/len(df)*100:.1f}%)")
    else:
        print(f"✅ ATR validity: {atr_valid:,}/{len(df):,} valid")
    
    # Feature count
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'label', 'forward_return']]
    
    if len(feature_cols) < 20:
        print(f"⚠️  WARNING: Only {len(feature_cols)} features available")
        print(f"   Minimum 30-50 features recommended")
    else:
        print(f"✅ Feature count: {len(feature_cols)}")
    
    # Show feature categories
    bid_ask = sum(1 for col in feature_cols if any(x in col.lower() for x in ['bid', 'ask', 'mid', 'spread']))
    indicators = len(feature_cols) - bid_ask - 5  # minus OHLCV
    
    print(f"\n📊 Feature Breakdown:")
    print(f"   Bid/Ask/Spread: {bid_ask}")
    print(f"   Technical Indicators: {indicators}")
    print(f"   Total Features: {len(feature_cols)}")
    
    # Date range
    print(f"\n📅 Data Range:")
    print(f"   Start: {df.index.min()}")
    print(f"   End: {df.index.max()}")
    print(f"   Duration: {duration_days} days")
    
    # Price stats
    print(f"\n💰 Price Statistics:")
    print(f"   Close range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   ATR mean: {df['ATR'].mean():.4f}")
    
    if 'spread' in df.columns:
        spread_to_atr = (df['spread'] / df['ATR']).median()
        print(f"   Spread/ATR ratio: {spread_to_atr:.4f}")
    
    print(f"\n✅ READY FOR TRAINING")
    return True


def main():
    parser = argparse.ArgumentParser(description='Pre-flight check before training')
    parser.add_argument('--timeframe', type=str, default=None,
                        help='Timeframe to check (5T, 15T, 1H, 4H)')
    parser.add_argument('--all', action='store_true',
                        help='Check all available timeframes')
    parser.add_argument('--base-path', type=str,
                        default="/Users/omar/Desktop/ML_model/ML_model/feature_store",
                        help='Base path to feature store')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PRE-FLIGHT CHECK - TRAINING READINESS")
    print("="*70)
    print(f"Base path: {args.base_path}")
    
    symbol = 'XAUUSD'
    
    if args.all:
        timeframes = ['5T', '15T', '30T', '1H', '4H']
    elif args.timeframe:
        timeframes = [args.timeframe]
    else:
        # Default to 1H
        timeframes = ['1H']
    
    results = {}
    for tf in timeframes:
        results[tf] = check_unified_file(symbol, tf, args.base_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} checks passed")
    
    for tf, result in results.items():
        status = '✅' if result else '❌'
        print(f"   {status} {symbol} {tf}")
    
    if passed == total:
        print(f"\n✅ ALL CHECKS PASSED - Ready to train!")
        print(f"\nRun:")
        for tf in timeframes:
            print(f"   python train_xauusd_unified.py --timeframe {tf} --quick-test")
    else:
        print(f"\n⚠️  {total - passed} checks failed")
        print(f"\nFix issues above, then run this check again")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()