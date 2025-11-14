#!/usr/bin/env python3
"""
SIGNAL GENERATION DIAGNOSTIC TOOL
===================================
Checks all components needed for signal generation and identifies issues.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

def check_environment():
    """Check environment variables."""
    print("=" * 60)
    print("1. ENVIRONMENT VARIABLES")
    print("=" * 60)

    required = {
        'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY'),
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_KEY': os.getenv('SUPABASE_KEY'),
    }

    all_present = True
    for key, value in required.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {'present' if value else 'MISSING'}")
        if not value:
            all_present = False

    return all_present

def check_imports():
    """Check if all required modules can be imported."""
    print("\n" + "=" * 60)
    print("2. MODULE IMPORTS")
    print("=" * 60)

    modules = [
        'requests',
        'pandas',
        'numpy',
        'pandas_ta',
        'supabase',
        'lightgbm',
        'xgboost',
        'sklearn',
        'balanced_model',
        'ensemble_predictor',
        'live_feature_utils',
        'news_filter',
        'market_costs',
        'execution_guardrails',
    ]

    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_ok = False

    return all_ok

def check_model_files():
    """Check if model files exist."""
    print("\n" + "=" * 60)
    print("3. MODEL FILES (XAUUSD + XAGUSD only)")
    print("=" * 60)

    # Only check XAUUSD and XAGUSD (active models)
    models = [
        ('XAGUSD', ['15T', '1H', '30T', '5T', '4H']),
        ('XAUUSD', ['15T', '1H', '30T', '5T']),
    ]

    all_ok = True
    for symbol, timeframes in models:
        for tf in timeframes:
            # Fixed path: models are in subdirectories with _PRODUCTION_READY suffix
            pkl_path = Path(f'models_production/{symbol}/{symbol}_{tf}_PRODUCTION_READY.pkl')
            json_path = Path(f'models_onnx/{symbol}/{symbol}_{tf}.json')

            pkl_ok = pkl_path.exists()
            json_ok = json_path.exists()

            if pkl_ok and json_ok:
                print(f"✅ {symbol} {tf:>4}: PKL + JSON present")
            else:
                print(f"❌ {symbol} {tf:>4}: PKL={'✅' if pkl_ok else '❌'} JSON={'✅' if json_ok else '❌'}")
                all_ok = False

    return all_ok

def check_ensemble_loading():
    """Check if ensemble predictors can be loaded."""
    print("\n" + "=" * 60)
    print("4. ENSEMBLE PREDICTOR LOADING (XAUUSD + XAGUSD only)")
    print("=" * 60)

    all_ok = True
    try:
        # Import after checking modules exist
        from ensemble_predictor import EnsemblePredictor

        # Only check active models
        symbols = ['XAGUSD', 'XAUUSD']

        for symbol in symbols:
            try:
                predictor = EnsemblePredictor(symbol)
                print(f"✅ {symbol}: {len(predictor.models)} models loaded")
            except Exception as e:
                print(f"❌ {symbol}: {e}")
                all_ok = False

    except Exception as e:
        print(f"❌ Cannot import EnsemblePredictor: {e}")
        all_ok = False

    return all_ok

def check_polygon_api():
    """Check if Polygon API is accessible."""
    print("\n" + "=" * 60)
    print("5. POLYGON API ACCESS")
    print("=" * 60)

    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ POLYGON_API_KEY not set - skipping API check")
        return False

    try:
        import requests
        from datetime import datetime, timedelta

        # Test with EURUSD (simple pair)
        ticker = 'C:EURUSD'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': api_key}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results_count = len(data.get('results', []))
            print(f"✅ Polygon API accessible: {results_count} bars fetched for EURUSD")
            return True
        else:
            print(f"❌ Polygon API returned {response.status_code}: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Polygon API check failed: {e}")
        return False

def check_supabase_access():
    """Check if Supabase is accessible."""
    print("\n" + "=" * 60)
    print("6. SUPABASE ACCESS")
    print("=" * 60)

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')

    if not url or not key:
        print("❌ SUPABASE_URL or SUPABASE_KEY not set - skipping Supabase check")
        return False

    try:
        from supabase import create_client

        supabase = create_client(url, key)

        # Try to read from live_signals table (correct table name)
        response = supabase.table('live_signals').select('*').limit(1).execute()

        print(f"✅ Supabase accessible: {len(response.data)} rows fetched from live_signals table")

        # Check most recent signal timestamps
        response = supabase.table('live_signals').select('symbol,timeframe,timestamp').order('timestamp', desc=True).limit(10).execute()

        if response.data:
            latest = response.data[0]
            latest_time = datetime.fromisoformat(latest['timestamp'].replace('Z', '+00:00'))
            age = datetime.now(timezone.utc) - latest_time

            print(f"   Most recent signal: {latest['symbol']} {latest['timeframe']} at {latest_time}")
            print(f"   Age: {age.total_seconds() / 60:.1f} minutes")

            if age.total_seconds() > 600:  # > 10 minutes
                print(f"   ⚠️  Signals are stale (>{age.total_seconds() / 60:.0f} minutes old)")

        return True

    except Exception as e:
        print(f"❌ Supabase check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic checks."""
    print("\n🔍 SIGNAL GENERATION DIAGNOSTIC")
    print(f"   Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"   Working directory: {Path.cwd()}\n")

    results = {
        'environment': check_environment(),
        'imports': check_imports(),
        'model_files': check_model_files(),
        'ensemble_loading': check_ensemble_loading(),
        'polygon_api': check_polygon_api(),
        'supabase': check_supabase_access(),
    }

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check.replace('_', ' ').title()}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Signal generation should work")
    else:
        print("❌ SOME CHECKS FAILED - Review errors above")
        print("\nCommon fixes:")
        print("  • Missing environment variables: Set in GitHub Secrets")
        print("  • Missing modules: Check requirements.txt installation")
        print("  • Model loading failures: Check BalancedModel import order")
        print("  • API failures: Check API keys and network connectivity")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
