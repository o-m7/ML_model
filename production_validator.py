#!/usr/bin/env python3
"""
PRODUCTION VALIDATOR
====================

Stress tests trading strategies for real-world deployment.
Tests sensitivity to costs, slippage, position limits, and edge cases.

Usage:
    python3 production_validator.py --config A --symbol XAUUSD --tf 15T
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime

# Import config from walk_forward_validator
import sys
sys.path.insert(0, str(Path(__file__).parent))
from walk_forward_validator import CONFIGS, load_data, create_labels


class StressTestConfig:
    """Stress test scenarios."""
    
    # Cost scenarios (multipliers of base cost)
    COST_SCENARIOS = {
        'best_case': {'commission': 0.5, 'slippage': 0.5},      # 50% of base
        'normal': {'commission': 1.0, 'slippage': 1.0},         # Base case
        'realistic': {'commission': 1.5, 'slippage': 2.0},      # 50% worse commission, 2x slippage
        'worst_case': {'commission': 3.0, 'slippage': 4.0},     # 3x commission, 4x slippage
        'extreme': {'commission': 5.0, 'slippage': 5.0}         # Extreme degradation
    }
    
    # Position limit scenarios
    POSITION_SCENARIOS = {
        'aggressive': 0.20,      # 20% max position
        'normal': 0.10,          # 10% max position
        'conservative': 0.05,    # 5% max position
        'very_conservative': 0.02 # 2% max position
    }


def run_stress_backtest(df: pd.DataFrame, model, features: List[str],
                       config: Dict, cost_multiplier: Dict, max_position_pct: float) -> Dict:
    """
    Run backtest with specified cost and position constraints.
    """
    initial_capital = 100000
    risk_pct = config['risk_pct']
    confidence_threshold = config['confidence']
    
    # Apply cost multipliers
    base_commission = 0.00001  # 0.001%
    base_slippage = 0.000005   # 0.0005%
    commission_pct = base_commission * cost_multiplier['commission']
    slippage_pct = base_slippage * cost_multiplier['slippage']
    
    # Get predictions
    X = df[features].fillna(0).values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Filter by confidence
    high_conf_mask = probabilities > confidence_threshold
    
    # Trading simulation
    equity = initial_capital
    equity_curve = [equity]
    trades = []
    peak_equity = equity
    max_drawdown = 0
    rejected_trades = 0
    
    for i in range(len(df)):
        if not high_conf_mask[i]:
            equity_curve.append(equity)
            continue
        
        # Entry parameters
        entry_price = df['close'].iloc[i]
        tp_price = df['tp_level'].iloc[i]
        sl_price = df['sl_level'].iloc[i]
        
        # Position sizing
        risk_amount = equity * risk_pct
        sl_distance = entry_price - sl_price
        
        if sl_distance <= 0:
            equity_curve.append(equity)
            continue
        
        position_size = risk_amount / sl_distance
        position_value = position_size * entry_price
        
        # Apply position limit
        max_position = equity * max_position_pct
        if position_value > max_position:
            position_size = max_position / entry_price
            position_value = max_position
            rejected_trades += 1
        
        # Get outcome from label
        if 'target' not in df.columns:
            continue
        
        label = df['target'].iloc[i]
        
        if label == 1:
            outcome = 'win'
            pnl = position_size * (tp_price - entry_price)
        else:
            outcome = 'loss'
            pnl = position_size * (sl_price - entry_price)
        
        # Apply costs
        commission = position_value * commission_pct
        slippage = position_value * slippage_pct
        net_pnl = pnl - commission - slippage
        
        # Update equity
        equity += net_pnl
        equity_curve.append(equity)
        
        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity
        max_drawdown = max(max_drawdown, drawdown)
        
        # Circuit breaker
        if max_drawdown > 0.10:  # 10% emergency stop
            break
        
        trades.append({
            'outcome': outcome,
            'pnl': net_pnl,
            'equity': equity,
            'commission': commission,
            'slippage': slippage
        })
    
    # Calculate metrics
    if len(trades) == 0:
        return {
            'trades': 0,
            'win_rate': 0,
            'total_return_pct': 0,
            'max_drawdown_pct': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'avg_commission': 0,
            'avg_slippage': 0,
            'rejected_trades': rejected_trades,
            'viable': False
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
    
    # Check if still viable
    viable = (
        total_return_pct > 10.0 and  # Still profitable
        max_drawdown < 0.06 and      # DD under 6%
        profit_factor > 1.5          # Still good PF
    )
    
    return {
        'trades': len(trades_df),
        'win_rate': win_rate * 100,
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown * 100,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'avg_commission': trades_df['commission'].mean(),
        'avg_slippage': trades_df['slippage'].mean(),
        'rejected_trades': rejected_trades,
        'viable': viable
    }


def stress_test_costs(symbol: str, timeframe: str, config_name: str) -> Dict:
    """Test strategy under different cost scenarios."""
    print(f"\n{'='*80}")
    print(f"COST STRESS TEST: {config_name}")
    print(f"{'='*80}\n")
    
    config = CONFIGS[config_name]
    
    # Load data
    df = load_data(symbol, timeframe)
    df = create_labels(df, config['tp_r'], config['sl_r'])
    
    # Use recent data only (last 20% for speed)
    test_size = int(len(df) * 0.20)
    df = df.iloc[-test_size:].copy().reset_index(drop=True)
    
    # Load model
    model_dir = Path("models") / symbol
    matching_models = []
    
    for model_file in model_dir.glob("*.pkl"):
        meta_file = model_file.parent / f"{model_file.stem}_meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if abs(meta['config']['tp_r'] - config['tp_r']) < 0.1:
                    matching_models.append((model_file, meta))
    
    if not matching_models:
        raise FileNotFoundError(f"No model found with TP={config['tp_r']}R")
    
    model_file, meta = max(matching_models, key=lambda x: x[0].stat().st_mtime)
    
    with open(model_file, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    features = model_package['features']
    
    # Test each cost scenario
    results = {}
    
    for scenario_name, cost_mult in StressTestConfig.COST_SCENARIOS.items():
        result = run_stress_backtest(
            df, model, features, config,
            cost_mult, 0.10  # Normal position limit
        )
        results[scenario_name] = result
        
        status = "✓" if result['viable'] else "✗"
        print(f"{status} {scenario_name.upper():<15} "
              f"Return: {result['total_return_pct']:>6.2f}% | "
              f"DD: {result['max_drawdown_pct']:>5.2f}% | "
              f"PF: {result['profit_factor']:>4.2f} | "
              f"Trades: {result['trades']:>4}")
    
    return results


def stress_test_position_limits(symbol: str, timeframe: str, config_name: str) -> Dict:
    """Test strategy under different position size limits."""
    print(f"\n{'='*80}")
    print(f"POSITION LIMIT STRESS TEST: {config_name}")
    print(f"{'='*80}\n")
    
    config = CONFIGS[config_name]
    
    # Load data
    df = load_data(symbol, timeframe)
    df = create_labels(df, config['tp_r'], config['sl_r'])
    
    # Use recent data only
    test_size = int(len(df) * 0.20)
    df = df.iloc[-test_size:].copy().reset_index(drop=True)
    
    # Load model
    model_dir = Path("models") / symbol
    matching_models = []
    
    for model_file in model_dir.glob("*.pkl"):
        meta_file = model_file.parent / f"{model_file.stem}_meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if abs(meta['config']['tp_r'] - config['tp_r']) < 0.1:
                    matching_models.append((model_file, meta))
    
    if not matching_models:
        raise FileNotFoundError(f"No model found with TP={config['tp_r']}R")
    
    model_file, meta = max(matching_models, key=lambda x: x[0].stat().st_mtime)
    
    with open(model_file, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    features = model_package['features']
    
    # Test each position limit scenario
    results = {}
    normal_costs = {'commission': 1.0, 'slippage': 1.0}
    
    for scenario_name, max_pos in StressTestConfig.POSITION_SCENARIOS.items():
        result = run_stress_backtest(
            df, model, features, config,
            normal_costs, max_pos
        )
        results[scenario_name] = result
        
        status = "✓" if result['viable'] else "✗"
        print(f"{status} {scenario_name.upper():<20} ({max_pos*100:>4.1f}% max) "
              f"Return: {result['total_return_pct']:>6.2f}% | "
              f"Rejected: {result['rejected_trades']:>3}")
    
    return results


def comprehensive_stress_test(symbol: str, timeframe: str, config_name: str):
    """Run all stress tests and generate report."""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PRODUCTION STRESS TEST")
    print(f"{'='*80}")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}")
    print(f"Configuration: {config_name} - {CONFIGS[config_name]['name']}")
    print(f"{'='*80}\n")
    
    # Run cost stress tests
    cost_results = stress_test_costs(symbol, timeframe, config_name)
    
    # Run position limit stress tests
    position_results = stress_test_position_limits(symbol, timeframe, config_name)
    
    # Analyze robustness
    print(f"\n{'='*80}")
    print("ROBUSTNESS ANALYSIS")
    print(f"{'='*80}\n")
    
    # Count viable scenarios
    cost_viable = sum(1 for r in cost_results.values() if r['viable'])
    total_cost_scenarios = len(cost_results)
    
    position_viable = sum(1 for r in position_results.values() if r['viable'])
    total_position_scenarios = len(position_results)
    
    print(f"Cost Scenarios Passed:     {cost_viable}/{total_cost_scenarios}")
    print(f"Position Scenarios Passed: {position_viable}/{total_position_scenarios}")
    
    # Overall assessment
    cost_pass_rate = cost_viable / total_cost_scenarios
    position_pass_rate = position_viable / total_position_scenarios
    
    overall_robust = cost_pass_rate >= 0.6 and position_pass_rate >= 0.75
    
    print(f"\n{'='*80}")
    if overall_robust:
        print("✓✓✓ STRATEGY IS PRODUCTION-READY ✓✓✓")
        print("\nThe strategy maintains profitability under:")
        print("  - Increased trading costs")
        print("  - Position size constraints")
        print("  - Real-world degradation scenarios")
    else:
        print("⚠ STRATEGY NEEDS OPTIMIZATION")
        print("\nIssues detected:")
        if cost_pass_rate < 0.6:
            print("  - Too sensitive to trading costs")
            print("    Recommendation: Wider TP/SL or higher confidence threshold")
        if position_pass_rate < 0.75:
            print("  - Relies on large position sizes")
            print("    Recommendation: Increase number of trades or risk per trade")
    print(f"{'='*80}\n")
    
    # Save results
    output_dir = Path("stress_test_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"stress_test_{config_name}_{timestamp}.json"
    
    results = {
        'config': config_name,
        'config_details': CONFIGS[config_name],
        'cost_stress_tests': cost_results,
        'position_stress_tests': position_results,
        'robustness': {
            'cost_pass_rate': cost_pass_rate,
            'position_pass_rate': position_pass_rate,
            'overall_robust': overall_robust
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved: {output_file}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Production Validator')
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--tf', type=str, default='15T')
    parser.add_argument('--config', type=str, required=True, choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--test', type=str, choices=['costs', 'positions', 'all'], default='all')
    
    args = parser.parse_args()
    
    try:
        if args.test == 'costs':
            stress_test_costs(args.symbol, args.tf, args.config)
        elif args.test == 'positions':
            stress_test_position_limits(args.symbol, args.tf, args.config)
        else:
            comprehensive_stress_test(args.symbol, args.tf, args.config)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

