"""Performance metrics calculation."""

import numpy as np
import pandas as pd
from typing import List, Dict


def calculate_metrics(trades: List[Dict]) -> Dict:
    """
    Calculate comprehensive trading metrics.
    
    Args:
        trades: List of trade dictionaries with 'pnl', 'return', etc.
        
    Returns:
        Dictionary of metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'total_return_pct': 0.0
        }
    
    # Extract PnL
    pnls = np.array([t['pnl'] for t in trades])
    returns = np.array([t.get('return_pct', 0) for t in trades])
    
    # Basic stats
    total_trades = len(trades)
    wins = pnls > 0
    losses = pnls < 0
    
    win_count = wins.sum()
    loss_count = losses.sum()
    
    # Profit Factor
    gross_profit = pnls[wins].sum() if win_count > 0 else 0
    gross_loss = abs(pnls[losses].sum()) if loss_count > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
    
    # Win Rate
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    # Sharpe Ratio (per trade)
    sharpe_ratio = (returns.mean() / returns.std()) if returns.std() > 0 else 0
    
    # Drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max()
    
    # Convert to percentage (assuming initial capital)
    initial_capital = 100000  # From config
    max_drawdown_pct = (max_drawdown / initial_capital * 100)
    
    # Total return
    total_return_pct = (cumulative[-1] / initial_capital * 100)
    
    # Expectancy
    avg_win = pnls[wins].mean() if win_count > 0 else 0
    avg_loss = abs(pnls[losses].mean()) if loss_count > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
    
    return {
        'total_trades': int(total_trades),
        'win_count': int(win_count),
        'loss_count': int(loss_count),
        'profit_factor': float(profit_factor),
        'win_rate': float(win_rate),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown_pct': float(max_drawdown_pct),
        'total_return_pct': float(total_return_pct),
        'expectancy': float(expectancy),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss)
    }


def calculate_sharpe_per_trade(returns: np.ndarray) -> float:
    """Calculate Sharpe ratio per trade."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std()


def check_benchmarks(metrics: Dict, config: Dict) -> Tuple[bool, List[str]]:
    """
    Check if metrics pass go-live benchmarks.
    
    Args:
        metrics: Performance metrics
        config: Benchmark configuration
        
    Returns:
        (passed, failures)
    """
    failures = []
    
    # Check each benchmark
    if metrics['profit_factor'] < config.get('min_profit_factor', 1.60):
        failures.append(
            f"PF {metrics['profit_factor']:.2f} < {config['min_profit_factor']}"
        )
    
    if metrics['max_drawdown_pct'] > config.get('max_drawdown_pct', 6.0):
        failures.append(
            f"DD {metrics['max_drawdown_pct']:.1f}% > {config['max_drawdown_pct']}%"
        )
    
    if metrics['sharpe_ratio'] < config.get('min_sharpe_per_trade', 0.25):
        failures.append(
            f"Sharpe {metrics['sharpe_ratio']:.2f} < {config['min_sharpe_per_trade']}"
        )
    
    if metrics['win_rate'] < config.get('min_win_rate', 52.0):
        failures.append(
            f"WR {metrics['win_rate']:.1f}% < {config['min_win_rate']}%"
        )
    
    if metrics['total_trades'] < config.get('min_trades_oos', 200):
        failures.append(
            f"Trades {metrics['total_trades']} < {config['min_trades_oos']}"
        )
    
    passed = len(failures) == 0
    
    return passed, failures

