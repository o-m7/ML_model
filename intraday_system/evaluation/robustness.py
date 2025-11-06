"""Robustness testing with stress scenarios."""

import numpy as np
from typing import List, Dict


def stress_test(
    trades: List[Dict],
    cost_multiplier: float = 1.25,
    latency_bars: int = 1,
    feature_noise_std: float = 0.02
) -> Dict:
    """
    Apply stress tests to trading results.
    
    Args:
        trades: Original trade list
        cost_multiplier: Multiply costs by this factor
        latency_bars: Entry/exit latency
        feature_noise_std: Add Gaussian noise to features
        
    Returns:
        Stressed metrics
    """
    from .metrics import calculate_metrics
    
    # Create stressed trades
    stressed_trades = []
    
    for trade in trades:
        stressed_trade = trade.copy()
        
        # Increase costs
        original_pnl = trade['pnl']
        commission = trade.get('commission', 0)
        slippage = trade.get('slippage', 0)
        
        # Apply higher costs
        stressed_commission = commission * cost_multiplier
        stressed_slippage = slippage * cost_multiplier
        
        # Adjust PnL
        cost_increase = (stressed_commission + stressed_slippage) - (commission + slippage)
        stressed_trade['pnl'] = original_pnl - cost_increase
        
        # Latency impact (approximate with slight degradation)
        latency_drag = original_pnl * 0.05 * latency_bars  # 5% per bar
        stressed_trade['pnl'] -= latency_drag
        
        stressed_trades.append(stressed_trade)
    
    # Calculate metrics on stressed trades
    stressed_metrics = calculate_metrics(stressed_trades)
    
    return stressed_metrics


def monte_carlo_resample(
    trades: List[Dict],
    n_simulations: int = 1000,
    block_size: int = 10
) -> Dict:
    """
    Monte Carlo resampling to test robustness.
    
    Args:
        trades: Trade list
        n_simulations: Number of simulations
        block_size: Block bootstrap size
        
    Returns:
        Statistics of resampled results
    """
    from .metrics import calculate_metrics
    
    pfs = []
    sharpes = []
    dds = []
    
    for _ in range(n_simulations):
        # Block bootstrap resample
        n_blocks = len(trades) // block_size
        resampled_trades = []
        
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, len(trades) - block_size)
            resampled_trades.extend(trades[start_idx:start_idx+block_size])
        
        # Calculate metrics
        metrics = calculate_metrics(resampled_trades)
        pfs.append(metrics['profit_factor'])
        sharpes.append(metrics['sharpe_ratio'])
        dds.append(metrics['max_drawdown_pct'])
    
    return {
        'pf_mean': np.mean(pfs),
        'pf_std': np.std(pfs),
        'pf_5th': np.percentile(pfs, 5),
        'pf_95th': np.percentile(pfs, 95),
        'sharpe_mean': np.mean(sharpes),
        'dd_mean': np.mean(dds),
        'dd_95th': np.percentile(dds, 95)
    }

