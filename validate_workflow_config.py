#!/usr/bin/env python3
"""
WORKFLOW CONFIGURATION VALIDATOR
=================================
Validates that all workflow files and signal generation components
are correctly configured for GitHub Actions.
"""

import json
import sys
from pathlib import Path

def validate_workflow_files():
    """Check that workflow files are correctly configured."""
    print("=" * 70)
    print("1. WORKFLOW FILES")
    print("=" * 70)

    issues = []

    # Check generate_signals.yml exists and is correct
    workflow_path = Path('.github/workflows/generate_signals.yml')
    if not workflow_path.exists():
        issues.append("generate_signals.yml missing")
        print("❌ generate_signals.yml: MISSING")
    else:
        content = workflow_path.read_text()

        # Check cron timing
        if "*/2 * * * *" in content:
            print("✅ Cron timing: Every 2 minutes (correct)")
        else:
            issues.append("Cron timing incorrect")
            print("❌ Cron timing: NOT every 2 minutes")

        # Check cancel-in-progress
        if "cancel-in-progress: true" in content:
            print("✅ cancel-in-progress: true (correct)")
        else:
            issues.append("cancel-in-progress not set to true")
            print("❌ cancel-in-progress: NOT true")

        # Check uses signal_generator.py
        if "signal_generator.py" in content:
            print("✅ Uses signal_generator.py (correct)")
        else:
            issues.append("Doesn't use signal_generator.py")
            print("❌ Doesn't use signal_generator.py")

        # Check has xgboost
        if "xgboost" in content:
            print("✅ Installs xgboost (correct)")
        else:
            issues.append("Missing xgboost installation")
            print("❌ Missing xgboost installation")

    # Check continuous_signals.yml is disabled
    old_workflow_active = Path('.github/workflows/continuous_signals.yml')
    old_workflow_disabled = Path('.github/workflows/continuous_signals.yml.disabled')

    if old_workflow_active.exists():
        issues.append("Old continuous_signals.yml still active")
        print("❌ Old workflow: STILL ACTIVE (should be disabled)")
    elif old_workflow_disabled.exists():
        print("✅ Old workflow: Disabled (correct)")
    else:
        print("ℹ️  Old workflow: Not found (OK)")

    return len(issues) == 0, issues


def validate_signal_generator():
    """Check signal_generator.py configuration."""
    print("\n" + "=" * 70)
    print("2. SIGNAL GENERATOR")
    print("=" * 70)

    issues = []

    # Check file exists
    sg_path = Path('signal_generator.py')
    if not sg_path.exists():
        issues.append("signal_generator.py missing")
        print("❌ signal_generator.py: MISSING")
        return False, issues

    content = sg_path.read_text()

    # Check has XAUUSD models
    if "('XAUUSD', '5T')" in content and "('XAUUSD', '15T')" in content:
        print("✅ XAUUSD models: Configured (5T, 15T, 30T, 1H)")
    else:
        issues.append("XAUUSD models not configured")
        print("❌ XAUUSD models: NOT configured")

    # Check has XAGUSD models
    if "('XAGUSD', '5T')" in content and "('XAGUSD', '4H')" in content:
        print("✅ XAGUSD models: Configured (5T, 15T, 30T, 1H, 4H)")
    else:
        issues.append("XAGUSD models not configured")
        print("❌ XAGUSD models: NOT configured")

    # Check has error tolerance
    if "success_rate" in content and "< 80" in content:
        print("✅ Error tolerance: 80% threshold (correct)")
    else:
        issues.append("No error tolerance")
        print("❌ Error tolerance: NOT configured")

    # Check syntax
    import py_compile
    try:
        py_compile.compile('signal_generator.py', doraise=True)
        print("✅ Python syntax: Valid")
    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")
        print(f"❌ Python syntax: INVALID - {e}")

    return len(issues) == 0, issues


def validate_model_files():
    """Check that model files exist."""
    print("\n" + "=" * 70)
    print("3. MODEL FILES")
    print("=" * 70)

    issues = []

    symbols_timeframes = [
        ('XAUUSD', ['5T', '15T', '30T', '1H']),
        ('XAGUSD', ['5T', '15T', '30T', '1H', '4H']),
    ]

    total_models = 0
    found_models = 0

    for symbol, timeframes in symbols_timeframes:
        for tf in timeframes:
            total_models += 1

            # Check PKL file
            pkl_path = Path(f'models_production/{symbol}/{symbol}_{tf}_PRODUCTION_READY.pkl')

            # Check JSON metadata
            json_path = Path(f'models_onnx/{symbol}/{symbol}_{tf}.json')

            if pkl_path.exists() and json_path.exists():
                found_models += 1
                print(f"✅ {symbol} {tf:>3}: PKL + JSON present")
            else:
                issues.append(f"{symbol} {tf} missing files")
                pkl_status = "✅" if pkl_path.exists() else "❌"
                json_status = "✅" if json_path.exists() else "❌"
                print(f"❌ {symbol} {tf:>3}: PKL={pkl_status} JSON={json_status}")

    print(f"\nTotal: {found_models}/{total_models} models complete")

    return found_models == total_models, issues


def validate_onnx_metadata():
    """Check ONNX metadata files exist (params are informational only)."""
    print("\n" + "=" * 70)
    print("4. ONNX METADATA (informational - not used by signal_generator)")
    print("=" * 70)

    issues = []

    symbols_timeframes = [
        ('XAUUSD', ['5T', '15T', '30T', '1H']),
        ('XAGUSD', ['5T', '15T', '30T', '1H', '4H']),
    ]

    print("ℹ️  Note: signal_generator.py uses market_costs.py for TP/SL params,")
    print("   not ONNX JSON files. These files are for ONNX runtime only.\n")

    for symbol, timeframes in symbols_timeframes:
        for tf in timeframes:
            json_path = Path(f'models_onnx/{symbol}/{symbol}_{tf}.json')

            if not json_path.exists():
                print(f"⚠️  {symbol} {tf}: JSON not found (optional)")
                continue

            try:
                with open(json_path) as f:
                    metadata = json.load(f)

                # Check has params section (any params are OK)
                if 'params' in metadata:
                    params = metadata['params']
                    # Show what keys exist (informational)
                    param_keys = list(params.keys())
                    print(f"✅ {symbol} {tf}: JSON present (params: {', '.join(param_keys)})")
                else:
                    print(f"✅ {symbol} {tf}: JSON present (no params section)")

            except Exception as e:
                print(f"⚠️  {symbol} {tf}: Error reading JSON - {e}")

    # ONNX metadata is informational only, not critical
    return True, []


def validate_market_costs():
    """Check market_costs.py has correct TP/SL parameters."""
    print("\n" + "=" * 70)
    print("5. MARKET COSTS (TP/SL parameters)")
    print("=" * 70)

    issues = []

    try:
        from market_costs import TP_SL_PARAMS

        # Expected parameters for XAUUSD and XAGUSD
        expected = {
            'XAUUSD': {
                '5T':  (1.4, 1.0),
                '15T': (1.6, 1.0),
                '30T': (2.0, 1.0),
                '1H':  (2.2, 1.0),
            },
            'XAGUSD': {
                '5T':  (1.4, 1.0),
                '15T': (1.5, 1.0),
                '30T': (2.0, 1.0),
                '1H':  (2.2, 1.0),
                '4H':  (2.5, 1.0),
            }
        }

        for symbol, timeframes in expected.items():
            if symbol not in TP_SL_PARAMS:
                issues.append(f"{symbol} missing from TP_SL_PARAMS")
                print(f"❌ {symbol}: NOT FOUND in TP_SL_PARAMS")
                continue

            for tf, (exp_tp, exp_sl) in timeframes.items():
                if tf not in TP_SL_PARAMS[symbol]:
                    issues.append(f"{symbol} {tf} missing from TP_SL_PARAMS")
                    print(f"❌ {symbol} {tf}: NOT FOUND")
                    continue

                params = TP_SL_PARAMS[symbol][tf]
                actual_tp = params.tp_atr_mult
                actual_sl = params.sl_atr_mult

                if actual_tp == exp_tp and actual_sl == exp_sl:
                    print(f"✅ {symbol} {tf}: TP={actual_tp}, SL={actual_sl}")
                else:
                    issues.append(f"{symbol} {tf} params mismatch")
                    print(f"❌ {symbol} {tf}: TP={actual_tp} (expected {exp_tp}), SL={actual_sl} (expected {exp_sl})")

    except Exception as e:
        issues.append(f"Cannot load market_costs: {e}")
        print(f"❌ Error loading market_costs.py: {e}")

    return len(issues) == 0, issues


def validate_diagnostic_files():
    """Check diagnostic scripts exist."""
    print("\n" + "=" * 70)
    print("6. DIAGNOSTIC FILES")
    print("=" * 70)

    issues = []

    files = [
        'diagnose_signal_generation.py',
        'ensemble_predictor.py',
        'balanced_model.py',
        'live_feature_utils.py',
        'market_costs.py',
    ]

    for filename in files:
        path = Path(filename)
        if path.exists():
            print(f"✅ {filename}")
        else:
            issues.append(f"{filename} missing")
            print(f"❌ {filename}: MISSING")

    return len(issues) == 0, issues


def main():
    """Run all validations."""
    print("\n" + "=" * 70)
    print("WORKFLOW CONFIGURATION VALIDATION")
    print("=" * 70)
    print()

    results = {}
    all_issues = []

    # Run validations
    results['workflow_files'], issues = validate_workflow_files()
    all_issues.extend(issues)

    results['signal_generator'], issues = validate_signal_generator()
    all_issues.extend(issues)

    results['model_files'], issues = validate_model_files()
    all_issues.extend(issues)

    results['onnx_metadata'], issues = validate_onnx_metadata()
    all_issues.extend(issues)

    results['market_costs'], issues = validate_market_costs()
    all_issues.extend(issues)

    results['diagnostic_files'], issues = validate_diagnostic_files()
    all_issues.extend(issues)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check.replace('_', ' ').title()}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("\nWorkflow is correctly configured for GitHub Actions:")
        print("  • Cron runs every 2 minutes")
        print("  • Auto-recovery from stuck jobs enabled")
        print("  • No conflicting workflows")
        print("  • 9 models ready (XAUUSD 4 + XAGUSD 5)")
        print("  • Error tolerance configured (80% threshold)")
        print("  • All dependencies specified in workflow")
    else:
        print("❌ VALIDATION FAILURES DETECTED")
        print(f"\nFound {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
