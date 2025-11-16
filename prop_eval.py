"""
Funded account / prop firm evaluation module.

Checks if backtest results meet typical prop firm rules and funded account criteria.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class PropEvalResults:
    """Results of prop firm evaluation."""

    # Rule checks
    passes_daily_dd: bool
    passes_overall_dd: bool
    passes_min_trading_days: bool
    passes_min_trades: bool
    passes_target_metrics: bool

    # Violations
    max_daily_dd_pct: float
    max_overall_dd_pct: float
    trading_days: int
    total_trades: int

    # Target metrics
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float

    # Overall result
    passes_evaluation: bool

    # Final stats
    final_equity: float
    total_pnl: float
    total_pnl_pct: float

    # Details
    daily_dd_violations: int
    overall_dd_violations: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Pretty print results."""
        s = "\n" + "="*60 + "\n"
        s += "PROP FIRM EVALUATION RESULTS\n"
        s += "="*60 + "\n\n"

        s += f"Overall Result:      {'✓ PASS' if self.passes_evaluation else '✗ FAIL'}\n"
        s += f"\n"

        s += "Rule Checks:\n"
        s += f"  Daily DD Rule:     {'✓' if self.passes_daily_dd else '✗'} (Max: {self.max_daily_dd_pct:.2f}%)\n"
        s += f"  Overall DD Rule:   {'✓' if self.passes_overall_dd else '✗'} (Max: {self.max_overall_dd_pct:.2f}%)\n"
        s += f"  Min Trading Days:  {'✓' if self.passes_min_trading_days else '✗'} ({self.trading_days} days)\n"
        s += f"  Min Trades:        {'✓' if self.passes_min_trades else '✗'} ({self.total_trades} trades)\n"
        s += f"  Target Metrics:    {'✓' if self.passes_target_metrics else '✗'}\n"
        s += f"\n"

        s += "Performance Metrics:\n"
        s += f"  Win Rate:          {self.win_rate*100:.2f}%\n"
        s += f"  Profit Factor:     {self.profit_factor:.2f}\n"
        s += f"  Max Drawdown:      {self.max_drawdown_pct:.2f}%\n"
        s += f"\n"

        s += "Final Results:\n"
        s += f"  Final Equity:      ${self.final_equity:,.2f}\n"
        s += f"  Total PnL:         ${self.total_pnl:,.2f} ({self.total_pnl_pct:.2f}%)\n"
        s += f"\n"

        if not self.passes_evaluation:
            s += "Violations:\n"
            if not self.passes_daily_dd:
                s += f"  Daily DD violations: {self.daily_dd_violations}\n"
            if not self.passes_overall_dd:
                s += f"  Overall DD violations: {self.overall_dd_violations}\n"

        s += "="*60 + "\n"

        return s


def calculate_daily_pnl(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate daily PnL from equity curve.

    Args:
        equity_curve: Equity time series (any frequency)

    Returns:
        Daily PnL series
    """
    # Resample to daily (take last value of each day)
    daily_equity = equity_curve.resample('D').last().ffill()

    # Calculate daily PnL
    daily_pnl = daily_equity.diff()

    return daily_pnl


def calculate_daily_drawdown(
    equity_curve: pd.Series,
    initial_capital: float
) -> pd.Series:
    """
    Calculate daily drawdown percentage.

    Args:
        equity_curve: Equity time series
        initial_capital: Starting capital

    Returns:
        Daily drawdown % series
    """
    # Resample to daily
    daily_equity = equity_curve.resample('D').last().ffill()

    # Calculate daily DD from start of day
    # For each day, calculate max loss during that day
    daily_start = equity_curve.resample('D').first().ffill()

    # DD = (current - day_start) / day_start * 100
    daily_dd_pct = []

    for date in daily_start.index:
        day_start_equity = daily_start.loc[date]

        # Get all equity values for this day
        day_equity = equity_curve[equity_curve.index.date == date.date()]

        if len(day_equity) == 0:
            daily_dd_pct.append(0.0)
            continue

        # Max drawdown during the day
        min_equity = day_equity.min()
        dd_pct = ((min_equity - day_start_equity) / day_start_equity) * 100

        daily_dd_pct.append(dd_pct)

    daily_dd_series = pd.Series(daily_dd_pct, index=daily_start.index)

    return daily_dd_series


def check_prop_firm_rules(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    config
) -> PropEvalResults:
    """
    Check if backtest meets prop firm / funded account rules.

    Args:
        equity_curve: Equity time series
        trade_log: Trade log DataFrame
        config: Configuration object

    Returns:
        PropEvalResults object
    """
    initial_capital = config.prop_eval.initial_capital
    max_daily_dd_pct = config.prop_eval.max_daily_dd_pct
    max_overall_dd_pct = config.prop_eval.max_overall_dd_pct
    min_trading_days = config.prop_eval.min_trading_days
    min_trades = config.prop_eval.min_trades

    target_win_rate = config.prop_eval.target_win_rate
    target_pf = config.prop_eval.target_profit_factor
    target_max_dd = config.prop_eval.target_max_dd

    # Calculate metrics
    from metrics import calculate_metrics, check_target_metrics

    metrics = calculate_metrics(equity_curve, trade_log, initial_capital)

    # Check target metrics
    target_checks = check_target_metrics(
        metrics,
        target_win_rate=target_win_rate,
        target_pf=target_pf,
        target_max_dd=target_max_dd
    )

    # Calculate daily drawdowns
    daily_dd = calculate_daily_drawdown(equity_curve, initial_capital)

    # Check daily DD violations
    daily_dd_violations = (daily_dd < -max_daily_dd_pct).sum()
    passes_daily_dd = daily_dd_violations == 0

    max_daily_dd = daily_dd.min()

    # Check overall DD
    overall_dd_pct = metrics.max_drawdown_pct
    overall_dd_violations = 1 if abs(overall_dd_pct) > max_overall_dd_pct else 0
    passes_overall_dd = overall_dd_violations == 0

    # Trading days
    if len(equity_curve) > 0:
        trading_days = len(equity_curve.resample('D').last())
    else:
        trading_days = 0

    passes_min_trading_days = trading_days >= min_trading_days

    # Total trades
    total_trades = len(trade_log)
    passes_min_trades = total_trades >= min_trades

    # Target metrics
    passes_target_metrics = target_checks['all_pass']

    # Overall pass/fail
    passes_evaluation = all([
        passes_daily_dd,
        passes_overall_dd,
        passes_min_trading_days,
        passes_min_trades,
        passes_target_metrics
    ])

    # Final stats
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    total_pnl = final_equity - initial_capital
    total_pnl_pct = (total_pnl / initial_capital) * 100

    return PropEvalResults(
        passes_daily_dd=passes_daily_dd,
        passes_overall_dd=passes_overall_dd,
        passes_min_trading_days=passes_min_trading_days,
        passes_min_trades=passes_min_trades,
        passes_target_metrics=passes_target_metrics,
        max_daily_dd_pct=max_daily_dd,
        max_overall_dd_pct=overall_dd_pct,
        trading_days=trading_days,
        total_trades=total_trades,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
        max_drawdown_pct=metrics.max_drawdown_pct,
        passes_evaluation=passes_evaluation,
        final_equity=final_equity,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        daily_dd_violations=daily_dd_violations,
        overall_dd_violations=overall_dd_violations
    )


if __name__ == "__main__":
    from config import get_default_config

    config = get_default_config()

    # Create sample equity curve (successful scenario)
    dates = pd.date_range('2020-01-01', periods=5000, freq='5min')
    np.random.seed(42)

    # Upward trending equity with realistic drawdowns
    equity = pd.Series(
        100000 + np.random.randn(5000).cumsum() * 100 + np.arange(5000) * 2,
        index=dates
    )

    # Sample trades
    trade_data = {
        'entry_time': pd.date_range('2020-01-01', periods=150, freq='1D'),
        'exit_time': pd.date_range('2020-01-01 12:00', periods=150, freq='1D'),
        'pnl': np.random.randn(150) * 100 + 50,  # Positive bias
        'r_multiple': np.random.randn(150) * 0.5 + 1.0  # Avg R > 1
    }
    trade_log = pd.DataFrame(trade_data)

    # Calculate daily DD
    print("Calculating daily drawdowns...")
    daily_dd = calculate_daily_drawdown(equity, 100000)
    print(f"Max daily DD: {daily_dd.min():.2f}%")
    print(f"Daily DD violations: {(daily_dd < -5.0).sum()}")

    # Check prop firm rules
    print("\n" + "="*60)
    print("Running prop firm evaluation...")
    results = check_prop_firm_rules(equity, trade_log, config)

    print(results)

    # Test with failing scenario
    print("\n" + "="*60)
    print("Testing with failing scenario (high drawdown)...")

    # Create equity with large drawdown
    equity_fail = equity.copy()
    equity_fail.iloc[2000:2500] -= 8000  # Large DD

    results_fail = check_prop_firm_rules(equity_fail, trade_log, config)
    print(results_fail)
