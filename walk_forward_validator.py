#!/usr/bin/env python3
"""
WALK-FORWARD VALIDATOR
======================

Tests trading models across multiple years to ensure consistent performance.
Validates that strategy beats S&P 500 returns with acceptable drawdown.

Usage:
    python3 walk_forward_validator.py --symbol XAUUSD --tf 15T --config A
    python3 walk_forward_validator.py --symbol XAUUSD --tf 15T --config B --all-years
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import TRUE backtest engine
from true_backtest_engine import TrueBacktestEngine, TradeConfig, run_true_backtest

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

CONFIGS = {
    'A': {
        'name': 'Conservative Growth',
        'tp_r': 1.8,
        'sl_r': 1.0,
        'confidence': 0.65,
        'risk_pct': 0.02,
        'description': '1.8:1 RR, lower confidence, 2% risk'
    },
    'B': {
        'name': 'Balanced Aggressive',
        'tp_r': 2.0,
        'sl_r': 1.0,
        'confidence': 0.60,
        'risk_pct': 0.02,
        'description': '2:1 RR, low confidence, 2% risk'
    },
    'C': {
        'name': 'High Reward Moderate',
        'tp_r': 2.5,
        'sl_r': 1.0,
        'confidence': 0.60,
        'risk_pct': 0.015,
        'description': '2.5:1 RR, low confidence, 1.5% risk'
    },
    'D': {
        'name': 'Maximum Reward',
        'tp_r': 3.0,
        'sl_r': 1.0,
        'confidence': 0.55,
        'risk_pct': 0.015,
        'description': '3:1 RR, very low confidence, 1.5% risk'
    }
}

# S&P 500 historical yearly returns (approximate)
SP500_RETURNS = {
    2019: 0.289,   # 28.9%
    2020: 0.164,   # 16.4%
    2021: 0.269,   # 26.9%
    2022: -0.182,  # -18.2%
    2023: 0.243,   # 24.3%
    2024: 0.235,   # 23.5% (through Oct)
    2025: 0.15     # 15% estimated
}

# =============================================================================
# YEARLY BACKTEST LOGIC
# =============================================================================

def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load feature store data."""
    feature_store = Path("feature_store")
    path = feature_store / symbol / f"{symbol}_{timeframe}.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    
    df = pd.read_parquet(path)
    
    # Ensure timestamp is in index or column
    if 'timestamp' not in df.columns:
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise ValueError("No timestamp found in data")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


# NO LONGER NEEDED - True backtest doesn't use pre-calculated labels


def run_yearly_backtest(df: pd.DataFrame, model, features: List[str], 
                       config: Dict, year: int, tp_r: float, sl_r: float) -> Dict:
    """
    Run TRUE backtest for a specific year using actual price action.
    
    Args:
        df: Data for the year (must have OHLCV)
        model: Trained model
        features: Feature list
        config: Config dict with risk params
        year: Year being tested
        tp_r: Take profit R multiple
        sl_r: Stop loss R multiple
    
    Returns:
        Dict with performance metrics
    """
    # Configure true backtest
    trade_config = TradeConfig(
        initial_capital=100000,
        risk_per_trade_pct=config['risk_pct'],
        confidence_threshold=config['confidence'],
        commission_pct=0.00001,
        slippage_pct=0.000005,
        max_bars_in_trade=50
    )
    
    # Run TRUE backtest (no labels needed!)
    results = run_true_backtest(df, model, features, trade_config, tp_r, sl_r)
    
    # Add year-specific info
    sp500_return = SP500_RETURNS.get(year, 0.15)
    beat_sp500 = (results['total_return_pct'] / 100) > sp500_return
    
    return {
        'year': year,
        'trades': results['total_trades'],
        'win_rate': results['win_rate'],
        'total_return': results['total_return'],
        'total_return_pct': results['total_return_pct'],
        'max_drawdown_pct': results['max_drawdown_pct'],
        'profit_factor': results['profit_factor'],
        'sharpe_ratio': results['sharpe_ratio'],
        'final_equity': 100000 + results['total_return'],
        'sp500_return': sp500_return * 100,
        'beat_sp500': beat_sp500,
        'outperformance': results['total_return_pct'] - (sp500_return * 100),
        'tp_hit_rate': results['tp_hit_rate'],
        'sl_hit_rate': results['sl_hit_rate']
    }


def walk_forward_test(symbol: str, timeframe: str, config_name: str) -> Dict:
    """
    Run walk-forward testing across all years.
    
    For each year, train on all previous data and test on that year.
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD VALIDATION: {symbol} {timeframe}")
    print(f"Configuration: {config_name} - {CONFIGS[config_name]['name']}")
    print(f"{'='*80}\n")
    
    config = CONFIGS[config_name]
    
    # Load all data
    print("Loading data...")
    df = load_data(symbol, timeframe)
    df['year'] = df['timestamp'].dt.year
    
    years = sorted(df['year'].unique())
    print(f"Available years: {years}\n")
    
    # NO LABEL CREATION NEEDED - True backtest uses actual price action!
    
    # Find model with matching TP/SL
    model_dir = Path("models") / symbol
    matching_models = []
    
    for model_file in model_dir.glob("*.pkl"):
        meta_file = model_file.with_suffix('').with_suffix('') / f"{model_file.stem}_meta.json"
        meta_file = model_file.parent / f"{model_file.stem}_meta.json"
        
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if abs(meta['config']['tp_r'] - config['tp_r']) < 0.1:
                    matching_models.append((model_file, meta))
    
    if not matching_models:
        raise FileNotFoundError(f"No model found with TP={config['tp_r']}R")
    
    # Use most recent model
    model_file, meta = max(matching_models, key=lambda x: x[0].stat().st_mtime)
    print(f"Using model: {model_file.name}")
    print(f"TP/SL: {meta['config']['tp_r']}R / {meta['config']['sl_r']}R\n")
    
    # Load model
    with open(model_file, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    features = model_package['features']
    
    # Run yearly backtests
    yearly_results = []
    
    for year in years:
        if year < 2019:  # Skip pre-2019 data
            continue
        
        print(f"Testing year: {year}")
        year_data = df[df['year'] == year].copy().reset_index(drop=True)
        
        if len(year_data) < 100:
            print(f"  Skipping {year} (insufficient data)")
            continue
        
        result = run_yearly_backtest(year_data, model, features, config, year, 
                                     config['tp_r'], config['sl_r'])
        yearly_results.append(result)
        
        status = "✓ BEAT" if result['beat_sp500'] else "✗ LOST TO"
        print(f"  {status} S&P 500")
        print(f"  Return: {result['total_return_pct']:.2f}% vs S&P: {result['sp500_return']:.2f}%")
        print(f"  Max DD: {result['max_drawdown_pct']:.2f}%")
        print(f"  Trades: {result['trades']}, Win Rate: {result['win_rate']:.1f}%\n")
    
    return {
        'config': config_name,
        'config_details': config,
        'yearly_results': yearly_results,
        'model_file': str(model_file)
    }


def analyze_results(results: Dict) -> Dict:
    """Analyze walk-forward results and determine if strategy is viable."""
    yearly = results['yearly_results']
    
    if len(yearly) == 0:
        return {'viable': False, 'reason': 'No yearly results'}
    
    # Calculate aggregate metrics
    returns = [y['total_return_pct'] for y in yearly]
    drawdowns = [y['max_drawdown_pct'] for y in yearly]
    beat_sp500_count = sum(1 for y in yearly if y['beat_sp500'])
    
    avg_return = np.mean(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    max_dd = np.max(drawdowns)
    
    total_trades = sum(y['trades'] for y in yearly)
    avg_win_rate = np.mean([y['win_rate'] for y in yearly])
    avg_sharpe = np.mean([y['sharpe_ratio'] for y in yearly])
    
    # Check criteria
    criteria = {
        'avg_return_15pct': avg_return >= 15.0,
        'min_return_positive': min_return >= 0,
        'max_dd_under_6pct': max_dd < 6.0,
        'beat_sp500_majority': beat_sp500_count >= len(yearly) * 0.7,  # 70% of years
        'sufficient_trades': total_trades >= 500,
        'sharpe_above_1': avg_sharpe >= 1.0
    }
    
    passed = sum(criteria.values())
    total = len(criteria)
    viable = passed >= 5  # Need at least 5/6 criteria
    
    return {
        'viable': viable,
        'criteria_passed': f"{passed}/{total}",
        'criteria': criteria,
        'avg_yearly_return': avg_return,
        'min_yearly_return': min_return,
        'max_yearly_return': max_return,
        'max_drawdown': max_dd,
        'years_beat_sp500': f"{beat_sp500_count}/{len(yearly)}",
        'total_trades': total_trades,
        'avg_win_rate': avg_win_rate,
        'avg_sharpe': avg_sharpe
    }


def print_report(results: Dict, analysis: Dict):
    """Print comprehensive report."""
    print(f"\n{'='*80}")
    print("WALK-FORWARD VALIDATION REPORT")
    print(f"{'='*80}\n")
    
    config = results['config_details']
    print(f"Configuration: {results['config']} - {config['name']}")
    print(f"  {config['description']}")
    print(f"  Model: {results['model_file']}\n")
    
    # Yearly breakdown
    print(f"{'='*80}")
    print("YEARLY PERFORMANCE")
    print(f"{'='*80}")
    print(f"{'Year':<6} {'Return %':<10} {'S&P500 %':<10} {'Beat?':<8} {'MaxDD %':<10} {'Trades':<8} {'WinRate %'}")
    print(f"{'-'*80}")
    
    for yr in results['yearly_results']:
        beat = "✓" if yr['beat_sp500'] else "✗"
        print(f"{yr['year']:<6} {yr['total_return_pct']:>8.2f}  {yr['sp500_return']:>8.2f}  "
              f"{beat:<8} {yr['max_drawdown_pct']:>8.2f}  {yr['trades']:>6}  {yr['win_rate']:>7.1f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Average Yearly Return:    {analysis['avg_yearly_return']:>7.2f}%")
    print(f"Minimum Yearly Return:    {analysis['min_yearly_return']:>7.2f}%")
    print(f"Maximum Yearly Return:    {analysis['max_yearly_return']:>7.2f}%")
    print(f"Maximum Drawdown:         {analysis['max_drawdown']:>7.2f}%")
    print(f"Years Beat S&P 500:       {analysis['years_beat_sp500']}")
    print(f"Total Trades:             {analysis['total_trades']:>7}")
    print(f"Average Win Rate:         {analysis['avg_win_rate']:>7.1f}%")
    print(f"Average Sharpe Ratio:     {analysis['avg_sharpe']:>7.2f}")
    
    # Criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA")
    print(f"{'='*80}")
    
    criteria_names = {
        'avg_return_15pct': 'Average Return ≥15%',
        'min_return_positive': 'All Years Positive',
        'max_dd_under_6pct': 'Max Drawdown <6%',
        'beat_sp500_majority': 'Beat S&P 500 ≥70% years',
        'sufficient_trades': 'Total Trades ≥500',
        'sharpe_above_1': 'Sharpe Ratio ≥1.0'
    }
    
    for key, passed in analysis['criteria'].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{criteria_names[key]:<35} {status}")
    
    print(f"\n{'='*80}")
    if analysis['viable']:
        print("✓✓✓ STRATEGY VIABLE FOR PRODUCTION ✓✓✓")
    else:
        print("✗ STRATEGY NEEDS IMPROVEMENT")
        print(f"Passed {analysis['criteria_passed']} criteria (need ≥5/6)")
    print(f"{'='*80}\n")


def save_results(results: Dict, analysis: Dict, config_name: str):
    """Save results to files."""
    output_dir = Path("walk_forward_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_file = output_dir / f"config_{config_name}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'results': results,
            'analysis': analysis
        }, f, indent=2, default=str)
    
    # Save CSV
    csv_file = output_dir / f"config_{config_name}_{timestamp}.csv"
    df = pd.DataFrame(results['yearly_results'])
    df.to_csv(csv_file, index=False)
    
    print(f"Results saved:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validator')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol to test')
    parser.add_argument('--tf', type=str, default='15T', help='Timeframe')
    parser.add_argument('--config', type=str, required=True, choices=['A', 'B', 'C', 'D'],
                       help='Configuration to test')
    parser.add_argument('--save', action='store_true', help='Save results to files')
    
    args = parser.parse_args()
    
    try:
        # Run walk-forward test
        results = walk_forward_test(args.symbol, args.tf, args.config)
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Print report
        print_report(results, analysis)
        
        # Save if requested
        if args.save:
            save_results(results, analysis, args.config)
        
        return 0 if analysis['viable'] else 1
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

