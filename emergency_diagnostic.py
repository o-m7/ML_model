"""
Emergency diagnostic to identify leakage and DD calculation bugs.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import your current system
try:
    from citadel_training_system_v3 import (
        DataLoader, FeatureEngineer, TripleBarrierLabeler,
        RiskMetrics, compute_equity_and_dd_adaptive, CONFIG
    )
    print("✅ Successfully imported V3.0 system\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_feature_leakage():
    """Test if features use current bar data (LEAKAGE)."""
    print("="*80)
    print("TEST 1: FEATURE LEAKAGE DETECTION")
    print("="*80)
    
    # Create simple test data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=500, freq='5min'),
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 101,
        'low': np.random.randn(500).cumsum() + 99,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500),
        'atr': np.random.rand(500) * 2 + 1,
        'ema_20': np.random.randn(500).cumsum() + 100,
        'ema_50': np.random.randn(500).cumsum() + 100,
        'ema_200': np.random.randn(500).cumsum() + 100,
        'sma_20': np.random.randn(500).cumsum() + 100,
        'sma_50': np.random.randn(500).cumsum() + 100,
        'rsi': np.random.rand(500) * 100,
        'bb_upper': np.random.randn(500).cumsum() + 102,
        'bb_lower': np.random.randn(500).cumsum() + 98,
        'hour': [i % 24 for i in range(500)]
    })
    
    # Add features
    df_features = FeatureEngineer.engineer_all_features(df.copy())
    
    # Test: Change bar 200's close dramatically
    test_idx = 200
    original_close = df.loc[test_idx, 'close']
    
    df_test = df.copy()
    df_test.loc[test_idx, 'close'] = original_close * 2.0  # Double it
    df_test_features = FeatureEngineer.engineer_all_features(df_test)
    
    # Check if features at bar 200 changed
    feature_cols = [c for c in df_features.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    leakage_found = False
    for col in feature_cols:
        if col not in df_test_features.columns:
            continue
            
        original_val = df_features.loc[test_idx, col]
        new_val = df_test_features.loc[test_idx, col]
        
        if pd.notna(original_val) and pd.notna(new_val):
            if abs(original_val - new_val) > 1e-6:
                print(f"🚨 LEAKAGE DETECTED in '{col}'!")
                print(f"   Original: {original_val:.6f}")
                print(f"   After changing current bar close: {new_val:.6f}")
                print(f"   → Feature changed when current bar changed!")
                leakage_found = True
                break
    
    if not leakage_found:
        print("✅ No leakage detected - features properly use historical data only")
    else:
        print("\n❌ CRITICAL: Features are using CURRENT bar data!")
        print("   This causes 85%+ win rates in backtest.")
        print("   FIX: Add .shift(1) to all OHLCV inputs in FeatureEngineer")
    
    print()


def test_triple_barrier_leakage():
    """Test if triple barrier uses current bar close (LEAKAGE)."""
    print("="*80)
    print("TEST 2: TRIPLE BARRIER LEAKAGE DETECTION")
    print("="*80)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] + [100] * 90,
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] + [101] * 90,
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108] + [99] * 90,
        'atr': [1.0] * 100,
        'open': [100] * 100,
        'volume': [1000] * 100
    })
    
    labels_original, _, _ = TripleBarrierLabeler.label(df, tp_mult=2.0, sl_mult=1.0, time_barrier=5, timeframe='5T')
    
    # Change bar 5's close dramatically
    df_test = df.copy()
    df_test.loc[5, 'close'] = 200  # Massive change
    
    labels_test, _, _ = TripleBarrierLabeler.label(df_test, tp_mult=2.0, sl_mult=1.0, time_barrier=5, timeframe='5T')
    
    # Check if label at bar 5 changed
    if labels_original.iloc[5] != labels_test.iloc[5]:
        print(f"🚨 LEAKAGE DETECTED in TripleBarrierLabeler!")
        print(f"   Original label at bar 5: {labels_original.iloc[5]}")
        print(f"   After changing bar 5 close: {labels_test.iloc[5]}")
        print(f"   → Label changed when current bar changed!")
        print("\n❌ CRITICAL: Triple barrier is using CURRENT bar close!")
        print("   FIX: Change entry_price = df['close'].iloc[i-1]")
    else:
        print("✅ No leakage detected - triple barrier uses historical data only")
    
    print()


def test_drawdown_calculation():
    """Test if drawdown calculation is correct."""
    print("="*80)
    print("TEST 3: DRAWDOWN CALCULATION")
    print("="*80)
    
    # Simple losing streak
    r_multiples = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])  # 5 consecutive 1R losses
    
    equity, max_dd_pct, max_dd_r, diagnostics = compute_equity_and_dd_adaptive(
        r_multiples,
        base_risk=0.01,
        enable_adaptive=False  # Disable to test pure calculation
    )
    
    print(f"Test case: 5 consecutive 1R losses at 1% risk per trade")
    print(f"Expected DD: ~4.9% (1 - 0.99^5)")
    print(f"Actual DD: {max_dd_pct:.1f}%")
    
    if max_dd_pct > 100:
        print(f"\n🚨 CRITICAL BUG: Drawdown is {max_dd_pct:.1f}% (>100%!)")
        print(f"   This is mathematically impossible.")
        print(f"   Equity curve: {equity}")
        print(f"   Diagnostics: {diagnostics}")
        print("\n   Possible causes:")
        print("   1. Equity calculation is wrong")
        print("   2. Risk compounding is broken")
        print("   3. Using wrong formula for DD")
    elif max_dd_pct < 4.5 or max_dd_pct > 5.5:
        print(f"\n⚠️  WARNING: DD calculation may be off")
        print(f"   Expected ~4.9%, got {max_dd_pct:.1f}%")
    else:
        print(f"✅ Drawdown calculation looks correct")
    
    print(f"\nEquity curve: {equity}")
    print()


def test_actual_predictions():
    """Test actual model predictions to see if they make sense."""
    print("="*80)
    print("TEST 4: PREDICTION REALISM CHECK")
    print("="*80)
    
    # Load real data
    try:
        df, metadata = DataLoader.load_timeframe_data('XAUUSD', '5T')
        print(f"✅ Loaded {len(df):,} bars of XAUUSD 5T data")
    except Exception as e:
        print(f"❌ Could not load data: {e}")
        return
    
    # Engineer features
    df_features = FeatureEngineer.engineer_all_features(df)
    
    # Take a sample
    sample_idx = len(df_features) // 2
    sample = df_features.iloc[sample_idx:sample_idx+100]
    
    print(f"\nChecking 100 bars starting at index {sample_idx}:")
    print(f"Sample date range: {sample['timestamp'].min()} to {sample['timestamp'].max()}")
    
    # Check for NaN features
    feature_cols = [c for c in sample.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    nan_counts = sample[feature_cols].isna().sum()
    
    if nan_counts.max() > 0:
        print(f"\n⚠️  WARNING: Found NaN values in features:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"   {col}: {nan_counts[col]} NaNs")
    
    # Check for extreme values
    extreme_cols = []
    for col in feature_cols:
        if sample[col].abs().max() > 1e6:
            extreme_cols.append(col)
    
    if extreme_cols:
        print(f"\n⚠️  WARNING: Found extreme values (>1e6) in features:")
        for col in extreme_cols[:5]:
            print(f"   {col}: max={sample[col].max():.2e}")
    
    print()


def test_trade_frequency():
    """Check if models are overtrading."""
    print("="*80)
    print("TEST 5: TRADE FREQUENCY CHECK")
    print("="*80)
    
    try:
        # Load the saved model
        import pickle
        model_path = Path("production_models/XAUUSD_5T_lightgbm.pkl")
        
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            return
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Model: {data['model_name']}")
        print(f"Threshold: {data['threshold']:.2f}")
        print(f"Total test trades: {data['performance']['total_trades']}")
        
        # Check if daily limit was applied
        max_trades_per_day = data.get('max_trades_per_day', 'Unknown')
        print(f"Max trades/day setting: {max_trades_per_day}")
        
        # Estimate trading days
        # If we don't have exact info, estimate from common test set size
        test_set_days = 30  # Rough estimate
        trades_per_day = data['performance']['total_trades'] / test_set_days
        
        print(f"Estimated trades/day: {trades_per_day:.1f}")
        
        if trades_per_day > 50:
            print(f"\n🚨 CRITICAL: {trades_per_day:.1f} trades/day is WAY too high!")
            print(f"   Daily trade limit may not be applied")
            print(f"   This causes overtrading and massive drawdowns")
        elif trades_per_day > 20:
            print(f"\n⚠️  WARNING: {trades_per_day:.1f} trades/day is high")
        else:
            print(f"\n✅ Trade frequency looks reasonable")
        
    except Exception as e:
        print(f"❌ Could not load model: {e}")
    
    print()


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*80)
    print("EMERGENCY DIAGNOSTIC - V3.0 SYSTEM")
    print("="*80)
    print("\nThis will test for:")
    print("1. Feature leakage (causes 85%+ WR)")
    print("2. Triple barrier leakage (causes 85%+ WR)")
    print("3. Drawdown calculation bugs (causes 500%+ DD)")
    print("4. Prediction realism")
    print("5. Trade frequency issues")
    print("\n" + "="*80 + "\n")
    
    test_feature_leakage()
    test_triple_barrier_leakage()
    test_drawdown_calculation()
    test_actual_predictions()
    test_trade_frequency()
    
    print("="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nIf you see 🚨 CRITICAL issues above, you MUST fix them before deployment.")
    print("\nRecommended actions:")
    print("1. If leakage detected → Apply V3.1 fixes (shift all features by 1 bar)")
    print("2. If DD calculation broken → Check compute_equity_and_dd_adaptive()")
    print("3. If overtrading → Verify daily trade limit filter is applied")
    print("\nSee the leakage_audit_report artifact for detailed fixes.")
    print()


if __name__ == '__main__':
    main()