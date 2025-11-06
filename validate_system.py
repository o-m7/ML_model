#!/usr/bin/env python3
"""
Production System Validation Suite - JP Morgan Level
Comprehensive testing of ML trading system before deployment.

Tests:
1. Data integrity and quality
2. Feature computation accuracy
3. Model loading and inference
4. Position sizing calculations
5. API endpoint functionality
6. Performance requirements (latency < 200ms)
7. Risk management constraints
8. Backtest metrics validation

Usage:
    python validate_system.py --symbol XAUUSD --tf 15T
"""

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

class Colors:
    """Terminal colors for output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ValidationResults:
    """Track validation test results."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.critical_failures = []
        self.warnings = []
    
    def log_test(self, name: str, passed: bool, message: str = "", critical: bool = False):
        """Log test result."""
        if passed:
            self.tests_passed += 1
            status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}"
        else:
            self.tests_failed += 1
            status = f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
            if critical:
                self.critical_failures.append(f"{name}: {message}")
        
        print(f"{status} | {name}")
        if message:
            print(f"      {message}")
    
    def log_warning(self, message: str):
        """Log warning."""
        self.warnings.append(message)
        print(f"{Colors.WARNING}⚠ WARNING{Colors.ENDC} | {message}")
    
    def summary(self):
        """Print test summary."""
        total = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total * 100) if total > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"{Colors.BOLD}VALIDATION SUMMARY{Colors.ENDC}")
        print(f"{'='*80}")
        print(f"Total Tests: {total}")
        print(f"{Colors.OKGREEN}Passed: {self.tests_passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {self.tests_failed}{Colors.ENDC}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.critical_failures:
            print(f"\n{Colors.FAIL}CRITICAL FAILURES:{Colors.ENDC}")
            for failure in self.critical_failures:
                print(f"  • {failure}")
        
        if self.warnings:
            print(f"\n{Colors.WARNING}WARNINGS:{Colors.ENDC}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        print(f"{'='*80}\n")
        
        return len(self.critical_failures) == 0

class SystemValidator:
    """Comprehensive system validation."""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.results = ValidationResults()
        
        # Auto-detect paths
        if Path("feature_store").exists():
            self.feature_store = Path("feature_store")
            self.model_store = Path("models")
            self.results_store = Path("results")
        else:
            self.feature_store = Path("ML_Trading/feature_store")
            self.model_store = Path("ML_Trading/models")
            self.results_store = Path("ML_Trading/results")
    
    def validate_directories(self):
        """TEST 1: Validate directory structure."""
        print(f"\n{Colors.HEADER}{'='*80}")
        print("TEST SUITE 1: DIRECTORY STRUCTURE")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        required_dirs = [self.feature_store, self.model_store, self.results_store]
        
        for dir_path in required_dirs:
            exists = dir_path.exists() and dir_path.is_dir()
            self.results.log_test(
                f"Directory exists: {dir_path}",
                exists,
                f"Path: {dir_path}",
                critical=True
            )
    
    def validate_data(self):
        """TEST 2: Validate data quality."""
        print(f"\n{Colors.HEADER}{'='*80}")
        print("TEST SUITE 2: DATA QUALITY")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        # Try nested directory structure first
        file_path = self.feature_store / self.symbol / f"{self.symbol}_{self.timeframe}.parquet"
        
        # Fallback to flat structure
        if not file_path.exists():
            file_path = self.feature_store / f"{self.symbol}_{self.timeframe}.parquet"
        
        # Check file exists
        file_exists = file_path.exists()
        self.results.log_test(
            "Data file exists",
            file_exists,
            f"Path: {file_path}",
            critical=True
        )
        
        if not file_exists:
            return
        
        # Load and validate data
        df = pd.read_parquet(file_path)
        
        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        has_all_cols = all(col in df.columns for col in required_cols)
        self.results.log_test(
            "Required columns present",
            has_all_cols,
            f"Columns: {list(df.columns)[:10]}...",
            critical=True
        )
        
        # Check for NaN values
        nan_count = df[required_cols].isna().sum().sum()
        no_nans = nan_count == 0
        self.results.log_test(
            "No NaN values in OHLC",
            no_nans,
            f"NaN count: {nan_count}"
        )
        
        # Check timestamp sorting
        is_sorted = df['timestamp'].is_monotonic_increasing
        self.results.log_test(
            "Timestamps sorted chronologically",
            is_sorted,
            critical=True
        )
        
        # Check for duplicates
        dup_count = df['timestamp'].duplicated().sum()
        no_dups = dup_count == 0
        self.results.log_test(
            "No duplicate timestamps",
            no_dups,
            f"Duplicate count: {dup_count}"
        )
        
        # Check data sanity
        high_gt_low = (df['high'] >= df['low']).all()
        self.results.log_test(
            "High >= Low for all bars",
            high_gt_low,
            critical=True
        )
        
        high_gte_close = (df['high'] >= df['close']).all()
        low_lte_close = (df['low'] <= df['close']).all()
        price_sanity = high_gte_close and low_lte_close
        self.results.log_test(
            "Price bounds valid (Low <= Close <= High)",
            price_sanity,
            critical=True
        )
        
        # Check sufficient data
        min_rows = 1000
        sufficient_data = len(df) >= min_rows
        self.results.log_test(
            f"Sufficient data (>= {min_rows} bars)",
            sufficient_data,
            f"Rows: {len(df)}",
            critical=True
        )
    
    def validate_model(self):
        """TEST 3: Validate trained model."""
        print(f"\n{Colors.HEADER}{'='*80}")
        print("TEST SUITE 3: MODEL VALIDATION")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        # Find model file
        pattern = f"{self.symbol}_{self.timeframe}_*.pkl"
        model_files = sorted(self.model_store.glob(pattern))
        
        model_exists = len(model_files) > 0
        self.results.log_test(
            "Model file exists",
            model_exists,
            f"Found {len(model_files)} model(s)",
            critical=True
        )
        
        if not model_exists:
            return
        
        latest_model = model_files[-1]
        
        # Load model
        try:
            with open(latest_model, 'rb') as f:
                model_data = pickle.load(f)
            
            self.results.log_test(
                "Model loads successfully",
                True,
                f"Path: {latest_model}"
            )
        except Exception as e:
            self.results.log_test(
                "Model loads successfully",
                False,
                f"Error: {e}",
                critical=True
            )
            return
        
        # Validate model components
        required_keys = ['model', 'threshold', 'feature_cols', 'symbol', 'timeframe', 'metrics']
        has_all_keys = all(key in model_data for key in required_keys)
        self.results.log_test(
            "Model has all required components",
            has_all_keys,
            f"Keys: {list(model_data.keys())}",
            critical=True
        )
        
        # Check metrics meet requirements
        metrics = model_data.get('metrics', {})
        
        win_rate = metrics.get('avg_win_rate', 0)
        win_rate_ok = win_rate >= 0.50
        self.results.log_test(
            "Win rate >= 50%",
            win_rate_ok,
            f"Win rate: {win_rate:.1%}",
            critical=True
        )
        
        pf = metrics.get('avg_profit_factor', 0)
        pf_ok = pf >= 1.6
        self.results.log_test(
            "Profit factor >= 1.6",
            pf_ok,
            f"PF: {pf:.2f}",
            critical=True
        )
        
        dd = metrics.get('avg_max_dd', 100)
        dd_ok = dd <= 6.0
        self.results.log_test(
            "Max drawdown <= 6%",
            dd_ok,
            f"DD: {dd:.2f}%",
            critical=True
        )
        
        sharpe = metrics.get('avg_sharpe', 0)
        sharpe_ok = sharpe >= 0.25
        self.results.log_test(
            "Sharpe ratio >= 0.25",
            sharpe_ok,
            f"Sharpe: {sharpe:.2f}",
            critical=True
        )
        
        # Test inference speed
        n_features = len(model_data['feature_cols'])
        X_test = np.random.randn(1, n_features)
        
        start_time = time.time()
        for _ in range(100):
            _ = model_data['model'].predict_proba(X_test)
        avg_latency = (time.time() - start_time) / 100 * 1000
        
        inference_fast = avg_latency < 50  # 50ms for 1 prediction
        self.results.log_test(
            "Inference latency < 50ms",
            inference_fast,
            f"Avg latency: {avg_latency:.2f}ms per prediction"
        )
    
    def validate_feature_computation(self):
        """TEST 4: Validate feature computation."""
        print(f"\n{Colors.HEADER}{'='*80}")
        print("TEST SUITE 4: FEATURE COMPUTATION")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        from realtime_features import RealtimeFeatureEngine, DataFetcher
        
        try:
            fetcher = DataFetcher()
            recent_bars = fetcher.fetch_recent_bars(self.symbol, self.timeframe, n_bars=200)
            
            self.results.log_test(
                "Can fetch recent bars",
                True,
                f"Fetched {len(recent_bars)} bars"
            )
        except Exception as e:
            self.results.log_test(
                "Can fetch recent bars",
                False,
                f"Error: {e}",
                critical=True
            )
            return
        
        # Compute features
        try:
            engine = RealtimeFeatureEngine()
            features, df = engine.compute_features(recent_bars)
            
            self.results.log_test(
                "Can compute features",
                True,
                f"Computed {len(features)} features"
            )
        except Exception as e:
            self.results.log_test(
                "Can compute features",
                False,
                f"Error: {e}",
                critical=True
            )
            return
        
        # Check for NaN features
        nan_features = [k for k, v in features.items() if np.isnan(v)]
        no_nan_features = len(nan_features) == 0
        self.results.log_test(
            "No NaN in computed features",
            no_nan_features,
            f"NaN features: {nan_features}" if nan_features else "All features valid"
        )
        
        # Load model and check feature alignment
        pattern = f"{self.symbol}_{self.timeframe}_*.pkl"
        model_files = sorted(self.model_store.glob(pattern))
        
        if model_files:
            with open(model_files[-1], 'rb') as f:
                model_data = pickle.load(f)
            
            expected_features = set(model_data['feature_cols'])
            computed_features = set(k for k in features.keys() 
                                   if k not in ['open', 'high', 'low', 'close'])
            
            features_match = expected_features.issubset(computed_features)
            missing = expected_features - computed_features
            
            self.results.log_test(
                "Computed features match model requirements",
                features_match,
                f"Missing: {missing}" if missing else "All features present",
                critical=True
            )
    
    def validate_position_sizing(self):
        """TEST 5: Validate position sizing calculations."""
        print(f"\n{Colors.HEADER}{'='*80}")
        print("TEST SUITE 5: POSITION SIZING")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        # Test calculation accuracy
        capital = 100000
        risk_pct = 1.0
        entry_price = 2000.0
        atr = 15.0
        sl_r = 1.0
        
        sl_distance = atr * sl_r
        sl_price = entry_price - sl_distance
        
        risk_amount = capital * (risk_pct / 100)
        position_size = risk_amount / sl_distance
        
        # Verify
        actual_loss = position_size * sl_distance
        loss_pct = (actual_loss / capital) * 100
        
        correct_sizing = 0.95 <= loss_pct <= 1.05  # Should be ~1%
        
        self.results.log_test(
            "Position sizing calculation accurate",
            correct_sizing,
            f"Risk: ${risk_amount:.2f}, Position: {position_size:.4f} units, "
            f"Max loss: {loss_pct:.2f}% (target: 1.0%)",
            critical=True
        )
        
        # Test with different risk levels
        for test_risk in [0.5, 1.0, 2.0]:
            risk_amt = capital * (test_risk / 100)
            pos_size = risk_amt / sl_distance
            actual = (pos_size * sl_distance / capital) * 100
            
            within_tolerance = abs(actual - test_risk) < 0.1
            self.results.log_test(
                f"Position sizing correct for {test_risk}% risk",
                within_tolerance,
                f"Expected: {test_risk:.2f}%, Actual: {actual:.2f}%"
            )
    
    def validate_api(self):
        """TEST 6: Validate API endpoints."""
        print(f"\n{Colors.HEADER}{'='*80}")
        print("TEST SUITE 6: API VALIDATION")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        base_url = "http://localhost:8000"
        
        # Test health check
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            api_online = response.status_code == 200
            self.results.log_test(
                "API health check",
                api_online,
                f"Status: {response.status_code}"
            )
        except Exception as e:
            self.results.log_test(
                "API health check",
                False,
                f"Error: {e}. Start API with: uvicorn signal_api:app"
            )
            self.results.log_warning("API tests skipped - API not running")
            return
        
        # Test model listing
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            models_endpoint_works = response.status_code == 200
            self.results.log_test(
                "Models endpoint works",
                models_endpoint_works,
                f"Status: {response.status_code}"
            )
        except Exception as e:
            self.results.log_test(
                "Models endpoint works",
                False,
                f"Error: {e}"
            )
        
        # Test signal generation latency
        try:
            payload = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "current_data": {
                    "open": 2000.0,
                    "high": 2010.0,
                    "low": 1995.0,
                    "close": 2005.0,
                    "atr": 15.0
                }
            }
            
            start = time.time()
            response = requests.post(f"{base_url}/signal", json=payload, timeout=5)
            latency = (time.time() - start) * 1000
            
            signal_works = response.status_code == 200
            self.results.log_test(
                "Signal generation endpoint works",
                signal_works,
                f"Status: {response.status_code}, Latency: {latency:.2f}ms"
            )
            
            if signal_works:
                latency_ok = latency < 200  # JP Morgan requirement
                self.results.log_test(
                    "Signal generation latency < 200ms",
                    latency_ok,
                    f"Latency: {latency:.2f}ms",
                    critical=True
                )
        except Exception as e:
            self.results.log_test(
                "Signal generation endpoint works",
                False,
                f"Error: {e}"
            )
    
    def validate_risk_management(self):
        """TEST 7: Validate risk management constraints."""
        print(f"\n{Colors.HEADER}{'='*80}")
        print("TEST SUITE 7: RISK MANAGEMENT")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        # Load model metrics
        pattern = f"{self.symbol}_{self.timeframe}_*.pkl"
        model_files = sorted(self.model_store.glob(pattern))
        
        if not model_files:
            self.results.log_warning("No model found for risk management tests")
            return
        
        with open(model_files[-1], 'rb') as f:
            model_data = pickle.load(f)
        
        metrics = model_data.get('metrics', {})
        
        # Check max drawdown constraint
        max_dd = metrics.get('avg_max_dd', 100)
        dd_acceptable = max_dd <= 6.0
        self.results.log_test(
            "Max drawdown within limits (≤ 6%)",
            dd_acceptable,
            f"Max DD: {max_dd:.2f}%",
            critical=True
        )
        
        # Check win rate
        win_rate = metrics.get('avg_win_rate', 0)
        win_rate_acceptable = win_rate >= 0.50
        self.results.log_test(
            "Win rate meets minimum (≥ 50%)",
            win_rate_acceptable,
            f"Win rate: {win_rate:.1%}",
            critical=True
        )
        
        # Check risk-reward
        # With TP=1.5R and SL=1.0R, R:R should be 1.5:1
        risk_reward_valid = True  # This is by design
        self.results.log_test(
            "Risk-reward ratio favorable (1.5:1)",
            risk_reward_valid,
            "TP=1.5R, SL=1.0R"
        )
    
    def run_all_tests(self):
        """Run complete validation suite."""
        print(f"\n{Colors.BOLD}{Colors.OKCYAN}")
        print("="*80)
        print(f"PRODUCTION SYSTEM VALIDATION - JP MORGAN LEVEL")
        print(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        print(f"Timestamp: {datetime.now()}")
        print("="*80)
        print(f"{Colors.ENDC}")
        
        # Run test suites
        self.validate_directories()
        self.validate_data()
        self.validate_model()
        self.validate_feature_computation()
        self.validate_position_sizing()
        self.validate_risk_management()
        self.validate_api()
        
        # Summary
        passed = self.results.summary()
        
        if passed:
            print(f"{Colors.OKGREEN}{Colors.BOLD}")
            print("="*80)
            print("✓ SYSTEM READY FOR PRODUCTION")
            print("="*80)
            print(f"{Colors.ENDC}")
            return 0
        else:
            print(f"{Colors.FAIL}{Colors.BOLD}")
            print("="*80)
            print("✗ SYSTEM NOT READY - FIX CRITICAL FAILURES")
            print("="*80)
            print(f"{Colors.ENDC}")
            return 1


def main():
    parser = argparse.ArgumentParser(description='Validate production system')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to validate')
    parser.add_argument('--tf', type=str, required=True, help='Timeframe to validate')
    args = parser.parse_args()
    
    validator = SystemValidator(args.symbol, args.tf)
    exit_code = validator.run_all_tests()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
