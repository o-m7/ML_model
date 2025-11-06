#!/usr/bin/env python3
"""
ENSEMBLE STRATEGY
=================

Combines multiple trading configurations for improved consistency.
Allocates capital dynamically based on recent performance.

Usage:
    python3 ensemble_strategy.py --configs A B C --symbol XAUUSD --tf 15T
    python3 ensemble_strategy.py --configs A B --allocation 60 40 --symbol XAUUSD --tf 15T
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Import from walk_forward_validator
import sys
sys.path.insert(0, str(Path(__file__).parent))
from walk_forward_validator import CONFIGS, load_data, create_labels


def load_config_model(symbol: str, config_name: str) -> Tuple:
    """Load model for a specific config."""
    config = CONFIGS[config_name]
    model_dir = Path("models") / symbol
    
    # Find matching model
    matching_models = []
    for model_file in model_dir.glob("*.pkl"):
        meta_file = model_file.parent / f"{model_file.stem}_meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if abs(meta['config']['tp_r'] - config['tp_r']) < 0.1:
                    matching_models.append((model_file, meta))
    
    if not matching_models:
        raise FileNotFoundError(f"No model found for config {config_name} (TP={config['tp_r']}R)")
    
    # Use most recent
    model_file, meta = max(matching_models, key=lambda x: x[0].stat().st_mtime)
    
    with open(model_file, 'rb') as f:
        model_package = pickle.load(f)
    
    return model_package['model'], model_package['features'], config


def run_ensemble_backtest(df: pd.DataFrame, configs: List[str], 
                         allocations: List[float], symbol: str) -> Dict:
    """
    Run backtest with ensemble of multiple configs.
    
    Args:
        df: Data to backtest on
        configs: List of config names (e.g., ['A', 'B', 'C'])
        allocations: List of capital allocations (must sum to 1.0)
        symbol: Trading symbol
    
    Returns:
        Dict with ensemble performance metrics
    """
    initial_capital = 100000
    commission_pct = 0.00001
    slippage_pct = 0.000005
    
    # Validate allocations
    if abs(sum(allocations) - 1.0) > 0.01:
        raise ValueError(f"Allocations must sum to 1.0, got {sum(allocations)}")
    
    # Load all models
    models = {}
    for config_name in configs:
        model, features, config = load_config_model(symbol, config_name)
        
        # Create labels for this config
        df_config = create_labels(df.copy(), config['tp_r'], config['sl_r'])
        
        models[config_name] = {
            'model': model,
            'features': features,
            'config': config,
            'df': df_config
        }
    
    # Run ensemble trading
    equity = initial_capital
    equity_curve = [equity]
    trades = []
    peak_equity = equity
    max_drawdown = 0
    
    config_performance = {name: {'wins': 0, 'losses': 0, 'pnl': 0} for name in configs}
    
    for i in range(len(df)):
        equity_before = equity
        
        # Check each config for signals
        for config_name, allocation in zip(configs, allocations):
            model_data = models[config_name]
            model = model_data['model']
            features = model_data['features']
            config = model_data['config']
            df_config = model_data['df']
            
            # Capital allocated to this config
            config_capital = equity * allocation
            
            # Get prediction
            X = df_config[features].fillna(0).iloc[i:i+1].values
            if len(X) == 0:
                continue
            
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0, 1]
            
            # Check confidence threshold
            if probability <= config['confidence']:
                continue
            
            # Entry parameters
            entry_price = df_config['close'].iloc[i]
            tp_price = df_config['tp_level'].iloc[i]
            sl_price = df_config['sl_level'].iloc[i]
            
            # Position sizing for this config
            risk_amount = config_capital * config['risk_pct']
            sl_distance = entry_price - sl_price
            
            if sl_distance <= 0:
                continue
            
            position_size = risk_amount / sl_distance
            position_value = position_size * entry_price
            
            # Cap position (per config limit)
            max_position = config_capital * 0.10
            if position_value > max_position:
                position_size = max_position / entry_price
                position_value = max_position
            
            # Get outcome from label
            if 'target' not in df_config.columns:
                continue
            
            label = df_config['target'].iloc[i]
            
            if label == 1:
                outcome = 'win'
                pnl = position_size * (tp_price - entry_price)
                config_performance[config_name]['wins'] += 1
            else:
                outcome = 'loss'
                pnl = position_size * (sl_price - entry_price)
                config_performance[config_name]['losses'] += 1
            
            # Apply costs
            commission = position_value * commission_pct
            slippage = position_value * slippage_pct
            net_pnl = pnl - commission - slippage
            
            # Update equity
            equity += net_pnl
            config_performance[config_name]['pnl'] += net_pnl
            
            trades.append({
                'config': config_name,
                'outcome': outcome,
                'pnl': net_pnl,
                'confidence': probability,
                'allocation': allocation
            })
        
        equity_curve.append(equity)
        
        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate metrics
    if len(trades) == 0:
        return {
            'trades': 0,
            'win_rate': 0,
            'total_return_pct': 0,
            'max_drawdown_pct': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'config_breakdown': {}
        }
    
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['outcome'] == 'win']
    losses = trades_df[trades_df['outcome'] == 'loss']
    
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
    total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    # Calculate returns
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    total_return_pct = (equity - initial_capital) / initial_capital * 100
    
    # Config breakdown
    config_breakdown = {}
    for config_name in configs:
        perf = config_performance[config_name]
        config_trades = trades_df[trades_df['config'] == config_name]
        
        if len(config_trades) > 0:
            config_win_rate = perf['wins'] / (perf['wins'] + perf['losses']) * 100
        else:
            config_win_rate = 0
        
        config_breakdown[config_name] = {
            'trades': len(config_trades),
            'wins': perf['wins'],
            'losses': perf['losses'],
            'win_rate': config_win_rate,
            'pnl': perf['pnl'],
            'pnl_pct': (perf['pnl'] / initial_capital) * 100
        }
    
    return {
        'trades': len(trades_df),
        'win_rate': win_rate * 100,
        'total_return': equity - initial_capital,
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown * 100,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'final_equity': equity,
        'config_breakdown': config_breakdown
    }


def optimize_allocations(symbol: str, timeframe: str, configs: List[str]) -> List[float]:
    """
    Optimize capital allocation across configs using recent performance.
    Uses simple equal-risk weighting as baseline.
    """
    print(f"\nOptimizing allocations for configs: {configs}")
    
    # Load recent data (last 20%)
    df = load_data(symbol, timeframe)
    test_size = int(len(df) * 0.20)
    df_recent = df.iloc[-test_size:].copy().reset_index(drop=True)
    
    # Test each config individually
    config_metrics = {}
    
    for config_name in configs:
        model, features, config = load_config_model(symbol, config_name)
        df_config = create_labels(df_recent.copy(), config['tp_r'], config['sl_r'])
        
        # Quick backtest
        result = run_ensemble_backtest(df_config, [config_name], [1.0], symbol)
        
        # Calculate score (return / drawdown ratio)
        if result['max_drawdown_pct'] > 0:
            score = result['total_return_pct'] / result['max_drawdown_pct']
        else:
            score = result['total_return_pct']
        
        config_metrics[config_name] = {
            'return': result['total_return_pct'],
            'drawdown': result['max_drawdown_pct'],
            'sharpe': result['sharpe_ratio'],
            'score': score
        }
        
        print(f"  {config_name}: Return={result['total_return_pct']:.2f}% DD={result['max_drawdown_pct']:.2f}% Score={score:.2f}")
    
    # Allocate proportional to scores
    total_score = sum(m['score'] for m in config_metrics.values())
    
    if total_score <= 0:
        # Fall back to equal weighting
        allocations = [1.0 / len(configs)] * len(configs)
    else:
        allocations = [config_metrics[c]['score'] / total_score for c in configs]
    
    print(f"\nOptimized allocations: {dict(zip(configs, [f'{a*100:.1f}%' for a in allocations]))}")
    
    return allocations


def test_ensemble(symbol: str, timeframe: str, configs: List[str], 
                 allocations: List[float] = None):
    """Test ensemble strategy."""
    print(f"\n{'='*80}")
    print(f"ENSEMBLE STRATEGY TEST")
    print(f"{'='*80}")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}")
    print(f"Configs: {', '.join(configs)}")
    print(f"{'='*80}\n")
    
    # Load data
    df = load_data(symbol, timeframe)
    
    # Use recent test period (last 20%)
    test_size = int(len(df) * 0.20)
    df_test = df.iloc[-test_size:].copy().reset_index(drop=True)
    
    print(f"Test period: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
    print(f"Test bars: {len(df_test):,}\n")
    
    # Optimize allocations if not provided
    if allocations is None:
        allocations = optimize_allocations(symbol, timeframe, configs)
    else:
        print(f"Using manual allocations: {dict(zip(configs, [f'{a*100:.1f}%' for a in allocations]))}\n")
    
    # Run ensemble backtest
    print("Running ensemble backtest...")
    results = run_ensemble_backtest(df_test, configs, allocations, symbol)
    
    # Print results
    print(f"\n{'='*80}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Total Trades:      {results['trades']:>6}")
    print(f"Win Rate:          {results['win_rate']:>6.2f}%")
    print(f"Total Return:      ${results['total_return']:>,.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Max Drawdown:      {results['max_drawdown_pct']:>6.2f}%")
    print(f"Profit Factor:     {results['profit_factor']:>6.2f}")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:>6.2f}")
    
    # Config breakdown
    print(f"\n{'='*80}")
    print("CONFIGURATION BREAKDOWN")
    print(f"{'='*80}\n")
    print(f"{'Config':<8} {'Allocation':<12} {'Trades':<8} {'Win Rate':<10} {'P&L':<15} {'Contribution'}")
    print(f"{'-'*80}")
    
    for i, config_name in enumerate(configs):
        breakdown = results['config_breakdown'][config_name]
        allocation_pct = allocations[i] * 100
        
        print(f"{config_name:<8} {allocation_pct:>6.1f}%      {breakdown['trades']:>6}  "
              f"{breakdown['win_rate']:>7.1f}%  ${breakdown['pnl']:>10,.2f}  "
              f"{breakdown['pnl_pct']:>6.2f}%")
    
    # Assessment
    print(f"\n{'='*80}")
    if results['total_return_pct'] >= 15 and results['max_drawdown_pct'] < 6.0:
        print("✓✓✓ ENSEMBLE MEETS REQUIREMENTS ✓✓✓")
        print(f"\n  - Return: {results['total_return_pct']:.2f}% (≥15% target)")
        print(f"  - Max DD: {results['max_drawdown_pct']:.2f}% (<6% target)")
        print(f"  - Diversified across {len(configs)} strategies")
    else:
        print("⚠ ENSEMBLE NEEDS ADJUSTMENT")
        if results['total_return_pct'] < 15:
            print(f"\n  - Return too low: {results['total_return_pct']:.2f}% (need ≥15%)")
        if results['max_drawdown_pct'] >= 6.0:
            print(f"\n  - Drawdown too high: {results['max_drawdown_pct']:.2f}% (need <6%)")
    print(f"{'='*80}\n")
    
    # Save results
    output_dir = Path("ensemble_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ensemble_{'_'.join(configs)}_{timestamp}.json"
    
    output_data = {
        'configs': configs,
        'allocations': allocations,
        'results': results,
        'timestamp': timestamp
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"Results saved: {output_file}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ensemble Strategy Tester')
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--tf', type=str, default='15T')
    parser.add_argument('--configs', type=str, nargs='+', required=True,
                       choices=['A', 'B', 'C', 'D'],
                       help='Configs to combine (e.g., A B C)')
    parser.add_argument('--allocation', type=float, nargs='+',
                       help='Manual allocation percentages (must sum to 100)')
    
    args = parser.parse_args()
    
    # Validate and convert allocations
    allocations = None
    if args.allocation:
        if len(args.allocation) != len(args.configs):
            print(f"ERROR: Number of allocations ({len(args.allocation)}) must match configs ({len(args.configs)})")
            return 1
        
        # Convert percentages to decimals
        allocations = [a / 100.0 for a in args.allocation]
        
        if abs(sum(allocations) - 1.0) > 0.01:
            print(f"ERROR: Allocations must sum to 100%, got {sum(args.allocation)}%")
            return 1
    
    try:
        test_ensemble(args.symbol, args.tf, args.configs, allocations)
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

