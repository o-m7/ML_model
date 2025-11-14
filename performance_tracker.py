#!/usr/bin/env python3
"""
LIVE PERFORMANCE TRACKING SYSTEM
==================================

Monitors actual trading performance vs expected benchmarks.

Features:
- Tracks signals from Supabase
- Calculates actual win rate, profit factor, returns
- Compares to backtest expectations
- Alerts on performance degradation
- Generates daily performance reports

Usage:
    python performance_tracker.py --days 7
    python performance_tracker.py --symbol XAGUSD --tf 30T
    python performance_tracker.py --all
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"⚠️  Supabase not available - using local data only ({type(e).__name__})")
    SUPABASE_AVAILABLE = False
    create_client = None  # Define placeholder


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    symbol: str
    timeframe: str
    total_signals: int
    closed_trades: int
    open_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    avg_return_per_trade: float
    max_drawdown_pct: float
    sharpe_ratio: float
    expected_win_rate: float
    expected_pf: float
    expected_return: float
    performance_ratio: float  # actual / expected
    status: str  # "excellent", "good", "marginal", "degraded"


@dataclass
class TradeRecord:
    """Individual trade record."""
    symbol: str
    timeframe: str
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    pnl: Optional[float]
    return_pct: Optional[float]
    status: str  # "open", "win", "loss"


class PerformanceTracker:
    """Tracks live trading performance."""

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """Initialize tracker."""
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')

        if SUPABASE_AVAILABLE and self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            self.online = True
        else:
            self.supabase = None
            self.online = False
            print("⚠️  Running in offline mode")

        # Load expected performance from ONNX metadata
        self.expected_performance = self._load_expected_performance()

    def _load_expected_performance(self) -> Dict:
        """Load expected performance from ONNX metadata files."""
        expected = {}

        for symbol in ['XAUUSD', 'XAGUSD']:
            expected[symbol] = {}

            onnx_dir = Path(f'models_onnx/{symbol}')
            if not onnx_dir.exists():
                continue

            for json_file in onnx_dir.glob('*.json'):
                with open(json_file) as f:
                    data = json.load(f)

                tf = data['timeframe']
                backtest_results = data.get('backtest_results', {})

                expected[symbol][tf] = {
                    'win_rate': backtest_results.get('win_rate', 0),
                    'profit_factor': backtest_results.get('profit_factor', 0),
                    'total_return_pct': backtest_results.get('total_return_pct', 0),
                    'max_drawdown_pct': backtest_results.get('max_drawdown_pct', 0),
                    'sharpe_ratio': backtest_results.get('sharpe_ratio', 0)
                }

        return expected

    def fetch_signals(self, symbol: Optional[str] = None, timeframe: Optional[str] = None,
                     days: int = 7) -> List[TradeRecord]:
        """Fetch signals from Supabase."""
        if not self.online:
            print("❌ Cannot fetch signals - offline mode")
            return []

        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            # Query signals
            query = self.supabase.table('trading_signals').select('*')

            if symbol:
                query = query.eq('symbol', symbol)
            if timeframe:
                query = query.eq('timeframe', timeframe)

            query = query.gte('created_at', start_date.isoformat())

            response = query.execute()

            if not response.data:
                return []

            # Convert to TradeRecord
            trades = []
            for signal in response.data:
                trade = TradeRecord(
                    symbol=signal['symbol'],
                    timeframe=signal['timeframe'],
                    direction=signal['direction'],
                    entry_price=signal.get('entry_price', 0),
                    entry_time=datetime.fromisoformat(signal['created_at'].replace('Z', '+00:00')),
                    exit_price=signal.get('exit_price'),
                    exit_time=datetime.fromisoformat(signal['exit_time']) if signal.get('exit_time') else None,
                    pnl=signal.get('pnl'),
                    return_pct=signal.get('return_pct'),
                    status=signal.get('status', 'open')
                )
                trades.append(trade)

            return trades

        except Exception as e:
            print(f"❌ Error fetching signals: {e}")
            return []

    def calculate_performance(self, trades: List[TradeRecord], symbol: str, timeframe: str) -> PerformanceMetrics:
        """Calculate performance metrics from trades."""

        if not trades:
            # Return empty metrics with expected values
            expected = self.expected_performance.get(symbol, {}).get(timeframe, {})
            return PerformanceMetrics(
                symbol=symbol,
                timeframe=timeframe,
                total_signals=0,
                closed_trades=0,
                open_trades=0,
                win_rate=0,
                profit_factor=0,
                total_return_pct=0,
                avg_return_per_trade=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                expected_win_rate=expected.get('win_rate', 0),
                expected_pf=expected.get('profit_factor', 0),
                expected_return=expected.get('total_return_pct', 0),
                performance_ratio=0,
                status='no_data'
            )

        # Separate open and closed trades
        closed_trades = [t for t in trades if t.status in ['win', 'loss']]
        open_trades = [t for t in trades if t.status == 'open']

        if not closed_trades:
            expected = self.expected_performance.get(symbol, {}).get(timeframe, {})
            return PerformanceMetrics(
                symbol=symbol,
                timeframe=timeframe,
                total_signals=len(trades),
                closed_trades=0,
                open_trades=len(open_trades),
                win_rate=0,
                profit_factor=0,
                total_return_pct=0,
                avg_return_per_trade=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                expected_win_rate=expected.get('win_rate', 0),
                expected_pf=expected.get('profit_factor', 0),
                expected_return=expected.get('total_return_pct', 0),
                performance_ratio=0,
                status='insufficient_data'
            )

        # Calculate metrics
        wins = [t for t in closed_trades if t.status == 'win']
        losses = [t for t in closed_trades if t.status == 'loss']

        win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0

        total_wins = sum(t.pnl for t in wins if t.pnl)
        total_losses = abs(sum(t.pnl for t in losses if t.pnl))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        returns = [t.return_pct for t in closed_trades if t.return_pct is not None]
        total_return_pct = sum(returns) if returns else 0
        avg_return = np.mean(returns) if returns else 0

        # Calculate drawdown
        cumulative_returns = np.cumsum(returns) if returns else [0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Calculate Sharpe ratio
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

        # Get expected performance
        expected = self.expected_performance.get(symbol, {}).get(timeframe, {})
        expected_return = expected.get('total_return_pct', 0)

        # Calculate performance ratio
        performance_ratio = total_return_pct / expected_return if expected_return > 0 else 0

        # Determine status
        if performance_ratio >= 0.8 and win_rate >= expected.get('win_rate', 50) * 0.9:
            status = 'excellent'
        elif performance_ratio >= 0.6 and win_rate >= expected.get('win_rate', 50) * 0.8:
            status = 'good'
        elif performance_ratio >= 0.4:
            status = 'marginal'
        else:
            status = 'degraded'

        return PerformanceMetrics(
            symbol=symbol,
            timeframe=timeframe,
            total_signals=len(trades),
            closed_trades=len(closed_trades),
            open_trades=len(open_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return_pct=total_return_pct,
            avg_return_per_trade=avg_return,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            expected_win_rate=expected.get('win_rate', 0),
            expected_pf=expected.get('profit_factor', 0),
            expected_return=expected_return,
            performance_ratio=performance_ratio,
            status=status
        )

    def generate_report(self, symbol: Optional[str] = None, timeframe: Optional[str] = None, days: int = 7):
        """Generate performance report."""

        print('=' * 100)
        print(f'LIVE PERFORMANCE TRACKING REPORT - Last {days} Days')
        print('=' * 100)
        print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Mode: {"Online (Supabase)" if self.online else "Offline"}')
        print('=' * 100)

        symbols_to_check = [symbol] if symbol else ['XAUUSD', 'XAGUSD']

        for sym in symbols_to_check:
            timeframes_to_check = [timeframe] if timeframe else ['5T', '15T', '30T', '1H']

            print(f'\n{sym}:')
            print('-' * 100)
            print('{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<12} {:<10}'.format(
                'TF', 'Signals', 'Closed', 'WR', 'PF', 'Return', 'Expected', 'Status'
            ))
            print('-' * 100)

            for tf in timeframes_to_check:
                trades = self.fetch_signals(sym, tf, days)
                metrics = self.calculate_performance(trades, sym, tf)

                # Status emoji
                status_emoji = {
                    'excellent': '✅',
                    'good': '🟢',
                    'marginal': '🟡',
                    'degraded': '🔴',
                    'no_data': '⚪',
                    'insufficient_data': '⚪'
                }

                print('{:<8} {:<10} {:<10} {:<10.1f}% {:<10.2f} {:<12.1f}% {:<12.1f}% {:<10}'.format(
                    tf,
                    metrics.total_signals,
                    metrics.closed_trades,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.total_return_pct,
                    metrics.expected_return,
                    f'{status_emoji.get(metrics.status, "⚪")} {metrics.status}'
                ))

        print('\n' + '=' * 100)
        print('STATUS LEGEND:')
        print('  ✅ excellent: Performance >= 80% of expected')
        print('  🟢 good:      Performance >= 60% of expected')
        print('  🟡 marginal:  Performance >= 40% of expected')
        print('  🔴 degraded:  Performance < 40% of expected')
        print('  ⚪ no_data:   No signals generated yet')
        print('=' * 100)


def main():
    parser = argparse.ArgumentParser(description='Track live trading performance')
    parser.add_argument('--symbol', type=str, help='Symbol (XAUUSD, XAGUSD)')
    parser.add_argument('--tf', type=str, help='Timeframe (5T, 15T, 30T, 1H)')
    parser.add_argument('--days', type=int, default=7, help='Days to analyze (default: 7)')
    parser.add_argument('--all', action='store_true', help='Show all symbols and timeframes')

    args = parser.parse_args()

    tracker = PerformanceTracker()

    if args.all:
        tracker.generate_report(days=args.days)
    else:
        tracker.generate_report(symbol=args.symbol, timeframe=args.tf, days=args.days)


if __name__ == '__main__':
    main()
