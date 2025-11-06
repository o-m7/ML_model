#!/usr/bin/env python3
"""
Quick test of RenTec-grade components.

Tests the audit framework, regime classifier, and hard gates
on actual XAUUSD data.

Usage:
    python test_rentec_components.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from qa.audit import DataAuditor, AuditConfig, deduplicate_timestamps, winsorize_features
from features.regime import RegimeClassifier, print_regime_report
from evaluation.gates import HardGates, GateThresholds, ForensicReport


def test_audit_framework():
    """Test data quality auditor."""
    print("\n" + "="*80)
    print("TEST 1: DATA QUALITY AUDIT")
    print("="*80)
    
    # Load XAUUSD 1H data
    data_path = Path("feature_store/XAUUSD/XAUUSD_1H.parquet")
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        print("   Please ensure feature_store/XAUUSD/XAUUSD_1H.parquet exists")
        return False
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} bars from {data_path}")
    
    # Run audit
    try:
        auditor = DataAuditor(AuditConfig())
        audit_results = auditor.audit_all(df, symbol='XAUUSD')
        print("\n✅ AUDIT PASSED")
        return True
    except Exception as e:
        print(f"\n❌ AUDIT FAILED: {e}")
        return False


def test_regime_classifier():
    """Test regime classification."""
    print("\n" + "="*80)
    print("TEST 2: REGIME CLASSIFICATION")
    print("="*80)
    
    # Load data
    data_path = Path("feature_store/XAUUSD/XAUUSD_1H.parquet")
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return False
    
    df = pd.read_parquet(data_path)
    
    # Add regime
    classifier = RegimeClassifier()
    df = classifier.add_regime(df)
    
    # Print stats
    print_regime_report(df, 'XAUUSD')
    
    # Check distribution
    regime_pcts = df['regime'].value_counts(normalize=True) * 100
    
    print("Regime Distribution:")
    for regime, pct in regime_pcts.items():
        print(f"  {regime}: {pct:.1f}%")
    
    # Verify regime column exists
    assert 'regime' in df.columns, "Missing 'regime' column"
    assert 'regime_confidence' in df.columns, "Missing 'regime_confidence' column"
    
    print("\n✅ REGIME CLASSIFICATION COMPLETE")
    return True


def test_hard_gates():
    """Test hard gates evaluation."""
    print("\n" + "="*80)
    print("TEST 3: HARD GATES EVALUATION")
    print("="*80)
    
    # Create mock backtest results (simulate XAUUSD-1H current performance)
    mock_results = {
        'profit_factor': 1.07,
        'max_drawdown_pct': 22.1,
        'sharpe_ratio': 0.21,
        'win_rate': 42.2,
        'total_trades': 185,
        'avg_win': 850,
        'avg_loss': -795,
        'largest_loss': -2100,
        'initial_capital': 100000,
        'sl_hit_rate': 67.5,
        'tp_hit_rate': 25.0,
        'timeout_rate': 7.5,
        'oos_duration_months': 6
    }
    
    # Create mock trades DataFrame
    mock_trades = pd.DataFrame({
        'timestamp': pd.date_range('2025-04-01', periods=185, freq='12H'),
        'pnl': np.random.normal(-50, 1000, 185)  # Simulated losing trades
    })
    
    # Evaluate with hard gates
    gates = HardGates(GateThresholds())
    passed, failures, report = gates.evaluate(
        mock_results, 
        'XAUUSD', 
        '1H',
        mock_trades
    )
    
    # Print report
    report.print_report()
    
    # Save forensic JSON
    forensics_dir = Path("forensics")
    forensics_dir.mkdir(exist_ok=True)
    
    report_path = forensics_dir / "XAUUSD_1H_test.json"
    report.to_json(report_path)
    print(f"\n📄 Forensic report saved to: {report_path}")
    
    # Verify failure detection
    assert not passed, "Should have failed gates with current metrics"
    assert len(failures) > 0, "Should have failure messages"
    assert len(report.root_causes) > 0, "Should have root causes"
    assert len(report.ranked_fixes) > 0, "Should have ranked fixes"
    
    print("\n✅ HARD GATES EVALUATION COMPLETE")
    return True


def test_integration():
    """Test full integration: audit → regime → gates."""
    print("\n" + "="*80)
    print("TEST 4: FULL INTEGRATION")
    print("="*80)
    
    # Load data
    data_path = Path("feature_store/XAUUSD/XAUUSD_1H.parquet")
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return False
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} bars")
    
    # Step 1: Audit
    print("\n[1/3] Running audit...")
    auditor = DataAuditor(AuditConfig())
    
    try:
        audit_results = auditor.audit_all(df, symbol='XAUUSD')
        print("  ✅ Audit passed")
    except Exception as e:
        print(f"  ❌ Audit failed: {e}")
        return False
    
    # Step 2: Add regime
    print("\n[2/3] Adding regime classification...")
    classifier = RegimeClassifier()
    df = classifier.add_regime(df)
    
    regime_stats = classifier.get_regime_stats(df)
    print(f"  Trend: {regime_stats['trend_pct']:.1f}%")
    print(f"  Range: {regime_stats['range_pct']:.1f}%")
    print(f"  Neutral: {regime_stats['neutral_pct']:.1f}%")
    
    # Step 3: Simulate training and evaluation
    print("\n[3/3] Simulating model evaluation with hard gates...")
    
    # Mock results
    mock_results = {
        'profit_factor': 1.07,
        'max_drawdown_pct': 22.1,
        'sharpe_ratio': 0.21,
        'win_rate': 42.2,
        'total_trades': 185,
        'avg_win': 850,
        'avg_loss': -795,
        'largest_loss': -2100,
        'initial_capital': 100000,
        'sl_hit_rate': 67.5,
        'oos_duration_months': 6
    }
    
    gates = HardGates()
    passed, failures, report = gates.evaluate(mock_results, 'XAUUSD', '1H')
    
    print(f"\n  Live Ready: {'✅ YES' if passed else '❌ NO'}")
    print(f"  Failures: {len(failures)}")
    print(f"  Recommended Fixes: {len(report.ranked_fixes)}")
    
    print("\n✅ FULL INTEGRATION TEST COMPLETE")
    return True


def main():
    """Run all tests."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     RenTec-Grade ML Trading System - Component Tests            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("Data Quality Audit", test_audit_framework),
        ("Regime Classification", test_regime_classifier),
        ("Hard Gates Evaluation", test_hard_gates),
        ("Full Integration", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:.<60} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED - Components ready for integration!")
        return 0
    else:
        print("\n⚠️  Some tests failed - review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

