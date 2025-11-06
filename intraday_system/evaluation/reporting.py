"""Generate performance reports."""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


def generate_report(
    symbol: str,
    timeframe: str,
    strategy: str,
    cv_results: List[Dict],
    oos_metrics: Dict,
    benchmarks_passed: bool,
    failures: List[str],
    output_dir: Path
):
    """
    Generate comprehensive performance report.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        strategy: Strategy name
        cv_results: Cross-validation fold results
        oos_metrics: Out-of-sample metrics
        benchmarks_passed: Whether benchmarks were met
        failures: List of benchmark failures
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report dictionary
    report = {
        'symbol': symbol,
        'timeframe': timeframe,
        'strategy': strategy,
        'status': 'READY' if benchmarks_passed else 'FAILED',
        'cv_summary': {
            'n_folds': len(cv_results),
            'mean_accuracy': float(np.mean([r['accuracy'] for r in cv_results])),
            'std_accuracy': float(np.std([r['accuracy'] for r in cv_results]))
        } if cv_results else {},
        'oos_metrics': oos_metrics,
        'benchmarks': {
            'passed': benchmarks_passed,
            'failures': failures
        }
    }
    
    # Save JSON report
    report_path = output_dir / f"{symbol}_{timeframe}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create simple text report
    text_report = f"""
{'='*80}
TRADING SYSTEM PERFORMANCE REPORT
{'='*80}

Symbol: {symbol}
Timeframe: {timeframe}
Strategy: {strategy}
Status: {'✅ READY FOR PRODUCTION' if benchmarks_passed else '❌ FAILED BENCHMARKS'}

{'='*80}
OUT-OF-SAMPLE METRICS (Last 6 months)
{'='*80}

Total Trades:      {oos_metrics.get('total_trades', 0)}
Win Rate:          {oos_metrics.get('win_rate', 0):.1f}%
Profit Factor:     {oos_metrics.get('profit_factor', 0):.2f}
Sharpe/Trade:      {oos_metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown:      {oos_metrics.get('max_drawdown_pct', 0):.1f}%
Total Return:      {oos_metrics.get('total_return_pct', 0):.1f}%
Expectancy:        ${oos_metrics.get('expectancy', 0):.2f}

{'='*80}
BENCHMARK ASSESSMENT
{'='*80}

"""
    
    if benchmarks_passed:
        text_report += "✅ ALL BENCHMARKS PASSED - Model ready for production\n"
    else:
        text_report += "❌ BENCHMARKS FAILED:\n"
        for failure in failures:
            text_report += f"  - {failure}\n"
    
    text_report += f"\n{'='*80}\n"
    
    # Save text report
    text_path = output_dir / f"{symbol}_{timeframe}_report.txt"
    with open(text_path, 'w') as f:
        f.write(text_report)
    
    return report_path

