#!/usr/bin/env python3
"""
Benchmark Validator - Ensures only passing models get deployed
"""

from typing import Dict, List, Tuple


class BenchmarkValidator:
    """Validates model performance against production benchmarks"""
    
    # Production benchmarks - STRICT REQUIREMENTS
    MIN_PROFIT_FACTOR = 1.6   # ⚠️ STRICT: Only elite models (was 1.05)
    MAX_DRAWDOWN_PCT = 6.0    # ⚠️ STRICT: Tight risk control (was 7.5%)
    MIN_SHARPE = 0.05         # Keep same
    MIN_WIN_RATE = 45.0       # ⚠️ STRICT: High win rate required (was 39%)
    
    MIN_TRADES_BY_TF = {
        "5T": 200,
        "15T": 150,
        "30T": 100,
        "1H": 60,
        "4H": 40
    }
    
    @classmethod
    def validate(cls, results: Dict, timeframe: str, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate model results against benchmarks
        
        Args:
            results: Dictionary with backtest results
            timeframe: Timeframe being validated
            strict: If True, ALL benchmarks must pass. If False, allow minor violations.
        
        Returns:
            (passes, failures) tuple
        """
        failures = []
        
        # Required keys
        required_keys = ['profit_factor', 'max_drawdown_pct', 'sharpe_ratio', 'win_rate', 'total_trades']
        missing_keys = [k for k in required_keys if k not in results]
        
        if missing_keys:
            failures.append(f"Missing required keys: {missing_keys}")
            return False, failures
        
        # Get minimum trade requirement
        min_trades = cls.MIN_TRADES_BY_TF.get(timeframe, 60)
        
        # Check each benchmark
        if results['profit_factor'] < cls.MIN_PROFIT_FACTOR:
            failures.append(f"PF {results['profit_factor']:.2f} < {cls.MIN_PROFIT_FACTOR}")
        
        if results['max_drawdown_pct'] > cls.MAX_DRAWDOWN_PCT:
            failures.append(f"DD {results['max_drawdown_pct']:.1f}% > {cls.MAX_DRAWDOWN_PCT}%")
        
        if results['sharpe_ratio'] < cls.MIN_SHARPE:
            failures.append(f"Sharpe {results['sharpe_ratio']:.2f} < {cls.MIN_SHARPE}")
        
        if results['win_rate'] < cls.MIN_WIN_RATE:
            failures.append(f"WR {results['win_rate']:.1f}% < {cls.MIN_WIN_RATE}%")
        
        if results['total_trades'] < min_trades:
            failures.append(f"Trades {results['total_trades']} < {min_trades}")
        
        # Strict mode: All must pass
        if strict:
            passes = len(failures) == 0
        # Lenient mode: Allow 1 minor violation (not PF or DD)
        else:
            critical_failures = [f for f in failures if 'PF' in f or 'DD' in f]
            passes = len(critical_failures) == 0 and len(failures) <= 1
        
        return passes, failures
    
    @classmethod
    def format_results(cls, results: Dict) -> str:
        """Format results for display"""
        return (
            f"PF={results.get('profit_factor', 0):.2f}, "
            f"DD={results.get('max_drawdown_pct', 0):.1f}%, "
            f"Sharpe={results.get('sharpe_ratio', 0):.2f}, "
            f"WR={results.get('win_rate', 0):.1f}%, "
            f"Trades={results.get('total_trades', 0)}"
        )
    
    @classmethod
    def print_validation(cls, symbol: str, timeframe: str, results: Dict, 
                        passes: bool, failures: List[str]) -> None:
        """Print validation results"""
        
        print(f"\n{'='*80}")
        print(f"🔍 BENCHMARK VALIDATION: {symbol} {timeframe}")
        print(f"{'='*80}")
        print(f"\n📊 Results: {cls.format_results(results)}")
        print(f"\n📋 Benchmarks:")
        print(f"   • Profit Factor ≥ {cls.MIN_PROFIT_FACTOR}")
        print(f"   • Max Drawdown ≤ {cls.MAX_DRAWDOWN_PCT}%")
        print(f"   • Sharpe Ratio ≥ {cls.MIN_SHARPE}")
        print(f"   • Win Rate ≥ {cls.MIN_WIN_RATE}%")
        print(f"   • Min Trades ≥ {cls.MIN_TRADES_BY_TF.get(timeframe, 60)}")
        
        if passes:
            print(f"\n✅ PASSED - Model meets all benchmarks")
        else:
            print(f"\n❌ FAILED - Benchmark violations:")
            for failure in failures:
                print(f"   • {failure}")
        
        print(f"{'='*80}\n")


def validate_model_file(model_path: str, strict: bool = True) -> bool:
    """
    Validate a pickled model file
    
    Args:
        model_path: Path to .pkl model file
        strict: Strict validation mode
    
    Returns:
        True if model passes benchmarks
    """
    import pickle
    from pathlib import Path
    
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return False
    
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        symbol = model_data.get('symbol', 'UNKNOWN')
        timeframe = model_data.get('timeframe', 'UNKNOWN')
        results = model_data.get('backtest_results', {})
        
        passes, failures = BenchmarkValidator.validate(results, timeframe, strict=strict)
        BenchmarkValidator.print_validation(symbol, timeframe, results, passes, failures)
        
        return passes
        
    except Exception as e:
        print(f"❌ Error validating {model_file}: {e}")
        return False


def validate_all_production_models(models_dir: str = "models_production", 
                                   strict: bool = True) -> Tuple[int, int]:
    """
    Validate all production models
    
    Args:
        models_dir: Directory containing model files
        strict: Strict validation mode
    
    Returns:
        (passed_count, failed_count) tuple
    """
    from pathlib import Path
    
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_path}")
        return 0, 0
    
    # Find all PRODUCTION_READY.pkl files
    model_files = list(models_path.rglob("*_PRODUCTION_READY.pkl"))
    
    if not model_files:
        print(f"⚠️  No production-ready models found in {models_path}")
        return 0, 0
    
    print(f"\n{'='*80}")
    print(f"🔍 VALIDATING {len(model_files)} PRODUCTION MODELS")
    print(f"{'='*80}\n")
    
    passed = 0
    failed = 0
    
    for model_file in sorted(model_files):
        if validate_model_file(model_file, strict=strict):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"   Total Models: {len(model_files)}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   Pass Rate: {passed/len(model_files)*100:.1f}%")
    print(f"{'='*80}\n")
    
    return passed, failed


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Validate specific model
        model_path = sys.argv[1]
        strict = "--lenient" not in sys.argv
        
        if validate_model_file(model_path, strict=strict):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Validate all models
        strict = "--lenient" not in sys.argv
        passed, failed = validate_all_production_models(strict=strict)
        
        if failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)

