"""
Performance metrics calculation module.

Implements comprehensive trading metrics aligned with industry standards.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Basic stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # PnL metrics
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Risk-adjusted metrics
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    mar_ratio: float

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_pct: float
    avg_drawdown: float
    max_drawdown_duration: int  # in bars

    # R-multiple metrics
    avg_r: float
    expectancy: float

    # Trading frequency
    trades_per_month: float
    avg_trade_duration: float  # in bars

    # Additional stats
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Pretty print metrics."""
        s = "\n" + "="*60 + "\n"
        s += "PERFORMANCE METRICS\n"
        s += "="*60 + "\n\n"

        s += f"Total Trades:        {self.total_trades}\n"
        s += f"Win Rate:            {self.win_rate*100:.2f}%\n"
        s += f"Profit Factor:       {self.profit_factor:.2f}\n"
        s += f"\n"

        s += f"Total PnL:           ${self.total_pnl:,.2f} ({self.total_pnl_pct:.2f}%)\n"
        s += f"Average Win:         ${self.avg_win:,.2f}\n"
        s += f"Average Loss:        ${self.avg_loss:,.2f}\n"
        s += f"Largest Win:         ${self.largest_win:,.2f}\n"
        s += f"Largest Loss:        ${self.largest_loss:,.2f}\n"
        s += f"\n"

        s += f"Max Drawdown:        {self.max_drawdown_pct:.2f}%\n"
        s += f"Sharpe Ratio:        {self.sharpe_ratio:.2f}\n"
        s += f"Sortino Ratio:       {self.sortino_ratio:.2f}\n"
        s += f"Calmar Ratio:        {self.calmar_ratio:.2f}\n"
        s += f"\n"

        s += f"Average R:           {self.avg_r:.2f}\n"
        s += f"Expectancy:          ${self.expectancy:,.2f}\n"
        s += f"\n"

        s += f"Trades/Month:        {self.trades_per_month:.1f}\n"
        s += f"Avg Trade Duration:  {self.avg_trade_duration:.1f} bars\n"

        s += "="*60 + "\n"

        return s


def calculate_metrics(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    initial_capital: float
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Time series of equity values
        trade_log: DataFrame with trade details
        initial_capital: Starting capital

    Returns:
        PerformanceMetrics object
    """
    if len(trade_log) == 0:
        # Return empty metrics if no trades
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            total_pnl=0.0, total_pnl_pct=0.0, avg_win=0.0, avg_loss=0.0,
            largest_win=0.0, largest_loss=0.0, profit_factor=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0, mar_ratio=0.0,
            max_drawdown=0.0, max_drawdown_pct=0.0, avg_drawdown=0.0,
            max_drawdown_duration=0, avg_r=0.0, expectancy=0.0,
            trades_per_month=0.0, avg_trade_duration=0.0,
            consecutive_wins=0, consecutive_losses=0, recovery_factor=0.0
        )

    # Basic trade stats
    total_trades = len(trade_log)
    winning_trades = len(trade_log[trade_log['pnl'] > 0])
    losing_trades = len(trade_log[trade_log['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    # PnL stats
    total_pnl = trade_log['pnl'].sum()
    total_pnl_pct = (total_pnl / initial_capital) * 100

    wins = trade_log[trade_log['pnl'] > 0]['pnl']
    losses = trade_log[trade_log['pnl'] < 0]['pnl']

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    largest_win = wins.max() if len(wins) > 0 else 0.0
    largest_loss = losses.min() if len(losses) > 0 else 0.0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # R-multiple stats
    avg_r = trade_log['r_multiple'].mean()
    expectancy = trade_log['pnl'].mean()

    # Drawdown analysis
    dd_stats = calculate_drawdown_stats(equity_curve)
    max_drawdown = dd_stats['max_drawdown']
    max_drawdown_pct = dd_stats['max_drawdown_pct']
    avg_drawdown = dd_stats['avg_drawdown']
    max_dd_duration = dd_stats['max_duration']

    # Risk-adjusted ratios
    sharpe = calculate_sharpe_ratio(equity_curve)
    sortino = calculate_sortino_ratio(equity_curve)
    calmar = calculate_calmar_ratio(total_pnl_pct, max_drawdown_pct)
    mar = calmar  # MAR ratio is same as Calmar

    # Trading frequency
    if len(trade_log) > 0:
        duration = (trade_log['exit_time'].max() - trade_log['entry_time'].min()).days
        trades_per_month = (total_trades / max(duration, 1)) * 30.0

        # Average trade duration
        trade_log['duration'] = (trade_log['exit_time'] - trade_log['entry_time']).dt.total_seconds() / 60  # minutes
        avg_trade_duration = trade_log['duration'].mean()
    else:
        trades_per_month = 0.0
        avg_trade_duration = 0.0

    # Consecutive wins/losses
    consecutive_stats = calculate_consecutive_stats(trade_log)

    # Recovery factor
    recovery_factor = abs(total_pnl / max_drawdown) if max_drawdown != 0 else 0.0

    return PerformanceMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        mar_ratio=mar,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        avg_drawdown=avg_drawdown,
        max_drawdown_duration=max_dd_duration,
        avg_r=avg_r,
        expectancy=expectancy,
        trades_per_month=trades_per_month,
        avg_trade_duration=avg_trade_duration,
        consecutive_wins=consecutive_stats['max_wins'],
        consecutive_losses=consecutive_stats['max_losses'],
        recovery_factor=recovery_factor
    )


def calculate_drawdown_stats(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate drawdown statistics.

    Args:
        equity_curve: Series of equity values

    Returns:
        Dict with drawdown metrics
    """
    if len(equity_curve) == 0:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'avg_drawdown': 0.0,
            'max_duration': 0
        }

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Drawdown in dollars and percentage
    drawdown = equity_curve - running_max
    drawdown_pct = (drawdown / running_max) * 100

    # Max drawdown
    max_dd = drawdown.min()
    max_dd_pct = drawdown_pct.min()

    # Average drawdown (only negative values)
    avg_dd = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0.0

    # Max drawdown duration
    # Find periods in drawdown
    in_drawdown = drawdown < 0
    drawdown_periods = []
    start = None

    for i, is_dd in enumerate(in_drawdown):
        if is_dd and start is None:
            start = i
        elif not is_dd and start is not None:
            drawdown_periods.append(i - start)
            start = None

    if start is not None:  # Still in drawdown at end
        drawdown_periods.append(len(in_drawdown) - start)

    max_duration = max(drawdown_periods) if len(drawdown_periods) > 0 else 0

    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'avg_drawdown': avg_dd,
        'max_duration': max_duration
    }


def calculate_sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        equity_curve: Equity time series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily, adjust for intraday)

    Returns:
        Sharpe ratio
    """
    if len(equity_curve) < 2:
        return 0.0

    returns = equity_curve.pct_change().dropna()

    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Annualize
    mean_return = returns.mean() * periods_per_year
    std_return = returns.std() * np.sqrt(periods_per_year)

    sharpe = (mean_return - risk_free_rate) / std_return

    return sharpe


def calculate_sortino_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility).

    Args:
        equity_curve: Equity time series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    if len(equity_curve) < 2:
        return 0.0

    returns = equity_curve.pct_change().dropna()

    if len(returns) == 0:
        return 0.0

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    mean_return = returns.mean() * periods_per_year
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)

    sortino = (mean_return - risk_free_rate) / downside_std

    return sortino


def calculate_calmar_ratio(
    annual_return_pct: float,
    max_drawdown_pct: float
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        annual_return_pct: Annualized return in percentage
        max_drawdown_pct: Max drawdown in percentage (negative)

    Returns:
        Calmar ratio
    """
    if max_drawdown_pct == 0:
        return 0.0

    calmar = annual_return_pct / abs(max_drawdown_pct)

    return calmar


def calculate_consecutive_stats(trade_log: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate consecutive wins/losses.

    Args:
        trade_log: DataFrame with trade PnL

    Returns:
        Dict with max consecutive wins and losses
    """
    if len(trade_log) == 0:
        return {'max_wins': 0, 'max_losses': 0}

    # Determine if each trade is win or loss
    is_win = (trade_log['pnl'] > 0).astype(int)

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for win in is_win:
        if win:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    return {
        'max_wins': max_wins,
        'max_losses': max_losses
    }


def check_target_metrics(
    metrics: PerformanceMetrics,
    target_win_rate: float = 0.60,
    target_pf: float = 1.5,
    target_max_dd: float = 6.0
) -> Dict[str, bool]:
    """
    Check if metrics meet target thresholds.

    Args:
        metrics: PerformanceMetrics object
        target_win_rate: Minimum win rate (0.60 = 60%)
        target_pf: Minimum profit factor
        target_max_dd: Maximum drawdown in %

    Returns:
        Dict with pass/fail for each metric
    """
    results = {
        'win_rate_pass': metrics.win_rate >= target_win_rate,
        'profit_factor_pass': metrics.profit_factor >= target_pf,
        'max_dd_pass': abs(metrics.max_drawdown_pct) <= target_max_dd,
        'all_pass': False
    }

    results['all_pass'] = all([
        results['win_rate_pass'],
        results['profit_factor_pass'],
        results['max_dd_pass']
    ])

    return results


if __name__ == "__main__":
    # Test metrics calculation
    import pandas as pd

    # Create sample equity curve
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    equity = pd.Series(
        100000 + np.random.randn(1000).cumsum() * 1000,
        index=dates
    )

    # Create sample trade log
    trade_data = {
        'entry_time': pd.date_range('2020-01-01', periods=100, freq='10D'),
        'exit_time': pd.date_range('2020-01-02', periods=100, freq='10D'),
        'pnl': np.random.randn(100) * 500,
        'r_multiple': np.random.randn(100) * 1.5
    }
    trade_log = pd.DataFrame(trade_data)

    # Calculate metrics
    metrics = calculate_metrics(equity, trade_log, 100000)

    print(metrics)

    # Check targets
    checks = check_target_metrics(metrics)
    print("\nTarget Checks:")
    for key, val in checks.items():
        print(f"  {key}: {val}")
