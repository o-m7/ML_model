#!/usr/bin/env python3
"""
MODEL-DRIVEN PROP-FIRM BACKTESTING ENGINE
===========================================
Production-ready backtester for ML trading models under realistic prop-firm constraints.

Challenge Rules:
- Initial Balance: $25,000
- Profit Target: +6% ($1,500 → $26,500)
- Max Drawdown: -4% ($1,000 → $24,000 hard stop)

Features:
- Bar-by-bar simulation with model predictions
- Realistic costs: spread, slippage, commission
- Risk-based position sizing
- Full trade lifecycle management
- Prop-firm pass/fail logic

Usage:
    from backtest_model import PropFirmBacktester

    backtester = PropFirmBacktester(price_df, predict_function)
    results = backtester.run()
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION - Adjustable Parameters
# ============================================================================

# Prop Firm Challenge Rules
INITIAL_BALANCE = 25_000.0      # Starting balance (USD)
PROFIT_TARGET_PCT = 0.06        # 6% profit target
MAX_DRAWDOWN_PCT = 0.04         # 4% max drawdown

# Trading Rules
RISK_PER_TRADE_PCT = 0.005      # 0.5% of equity risked per trade
MAX_POSITIONS = 1               # Maximum concurrent positions
STOP_DISTANCE_POINTS = 300      # Stop loss distance (points) for position sizing

# Trading Costs
COMMISSION_PER_LOT = 7.0        # Round-turn commission per 1.0 lot (USD)
SLIPPAGE_POINTS_MEAN = 0.0      # Mean slippage (points)
SLIPPAGE_POINTS_STD = 2.0       # Std dev of slippage (points)

# Symbol-Specific Configuration
SYMBOL_CONFIG = {
    "XAUUSD": {
        "spread_points": 20.0,   # Bid-ask spread (points)
        "point_value": 0.10,     # USD per point per 1.0 lot
        "min_lot": 0.01,         # Minimum position size
        "max_lot": 10.0,         # Maximum position size
    },
    "XAGUSD": {
        "spread_points": 2.0,
        "point_value": 0.01,
        "min_lot": 0.01,
        "max_lot": 10.0,
    },
    "EURUSD": {
        "spread_points": 1.5,
        "point_value": 10.0,     # $10 per pip
        "min_lot": 0.01,
        "max_lot": 10.0,
    },
    "GBPUSD": {
        "spread_points": 2.0,
        "point_value": 10.0,
        "min_lot": 0.01,
        "max_lot": 10.0,
    },
}

# Default config for unknown symbols
DEFAULT_SYMBOL_CONFIG = {
    "spread_points": 2.0,
    "point_value": 10.0,
    "min_lot": 0.01,
    "max_lot": 10.0,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    direction: int          # +1 for long, -1 for short
    entry_price: float      # Price after spread/slippage
    lots: float             # Position size in lots
    entry_time: datetime
    entry_bar_index: int

    def __post_init__(self):
        """Validate position."""
        if self.direction not in [-1, 1]:
            raise ValueError(f"Direction must be +1 or -1, got {self.direction}")
        if self.lots <= 0:
            raise ValueError(f"Lots must be > 0, got {self.lots}")


@dataclass
class ClosedTrade:
    """Represents a closed trade with full details."""
    symbol: str
    direction: int          # +1 for long, -1 for short
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    lots: float
    gross_pnl: float        # PnL before costs
    spread_cost: float
    commission: float
    slippage_cost: float
    net_pnl: float          # Final PnL after all costs
    equity_before: float
    equity_after: float
    entry_bar_index: int
    exit_bar_index: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            "symbol": self.symbol,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "lots": self.lots,
            "gross_pnl": self.gross_pnl,
            "spread_cost": self.spread_cost,
            "commission": self.commission,
            "slippage_cost": self.slippage_cost,
            "net_pnl": self.net_pnl,
            "equity_before": self.equity_before,
            "equity_after": self.equity_after,
            "bars_held": self.exit_bar_index - self.entry_bar_index,
        }


@dataclass
class BacktestResults:
    """Complete backtest results."""
    passed: bool
    final_equity: float
    total_pnl: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    peak_equity: float
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_costs: float
    trades: List[ClosedTrade]
    equity_curve: pd.DataFrame
    failure_reason: Optional[str] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_symbol_config(symbol: str) -> Dict:
    """Get configuration for a symbol."""
    return SYMBOL_CONFIG.get(symbol.upper(), DEFAULT_SYMBOL_CONFIG)


def calculate_position_size(
    equity: float,
    symbol: str,
    stop_distance_points: float = STOP_DISTANCE_POINTS,
    risk_pct: float = RISK_PER_TRADE_PCT
) -> float:
    """
    Calculate position size in lots based on risk management.

    Formula:
        risk_amount = equity * risk_pct
        point_value = config["point_value"]
        lots = risk_amount / (stop_distance_points * point_value)

    Example for XAUUSD:
        equity = $25,000
        risk_pct = 0.005 (0.5%)
        risk_amount = $125
        stop_distance = 300 points
        point_value = $0.10/point
        lots = $125 / (300 * $0.10) = 125 / 30 = 4.17 lots
    """
    config = get_symbol_config(symbol)

    risk_amount = equity * risk_pct
    point_value = config["point_value"]

    # Calculate lots
    lots = risk_amount / (stop_distance_points * point_value)

    # Apply limits
    lots = max(config["min_lot"], lots)
    lots = min(config["max_lot"], lots)

    # Round to 2 decimals
    lots = round(lots, 2)

    return lots


def apply_spread_and_slippage(
    price: float,
    symbol: str,
    direction: int,
    is_entry: bool
) -> Tuple[float, float, float]:
    """
    Apply spread and slippage to a price.

    Args:
        price: Raw price from bar
        symbol: Trading symbol
        direction: +1 for long, -1 for short
        is_entry: True for entry, False for exit

    Returns:
        (adjusted_price, spread_cost, slippage_cost)

    Spread:
        - Long entry: pay ask (price + half_spread)
        - Long exit: receive bid (price - half_spread)
        - Short entry: pay bid (price - half_spread)
        - Short exit: receive ask (price + half_spread)

    Slippage:
        - Always applied AGAINST the trader
        - Random from N(mean, std)
    """
    config = get_symbol_config(symbol)
    spread_points = config["spread_points"]
    point_value = config["point_value"]

    # Spread: half applied now, half on exit
    half_spread_points = spread_points / 2.0

    # Sample slippage (always unfavorable)
    slippage_points = abs(np.random.normal(SLIPPAGE_POINTS_MEAN, SLIPPAGE_POINTS_STD))

    # Convert points to price
    # For XAUUSD: 1 point = $0.10, so to get price change: points * point_value / point_value
    # Actually for metals, 1 point IS the price unit
    # For XAUUSD at $2680: 20 points = 20 * $0.10 worth = $2 of spread

    # Calculate price adjustments
    if symbol.upper() in ["XAUUSD", "XAGUSD"]:
        # Metals: points are price units (1 point = min price fluctuation)
        # XAUUSD: 1 point = $0.10 price change
        spread_price_change = half_spread_points * point_value / point_value  # This equals half_spread_points
        slippage_price_change = slippage_points * point_value / point_value
    else:
        # Forex: points are pips
        spread_price_change = half_spread_points * 0.0001  # 1 pip = 0.0001
        slippage_price_change = slippage_points * 0.0001

    # Apply based on direction and entry/exit
    if is_entry:
        if direction == 1:  # Long entry: pay ask + slippage
            adjusted_price = price + spread_price_change + slippage_price_change
        else:  # Short entry: receive bid - slippage
            adjusted_price = price - spread_price_change - slippage_price_change
    else:
        if direction == 1:  # Long exit: receive bid - slippage
            adjusted_price = price - spread_price_change - slippage_price_change
        else:  # Short exit: pay ask + slippage
            adjusted_price = price + spread_price_change + slippage_price_change

    # Costs in USD (for 1 lot)
    spread_cost_usd = half_spread_points * point_value
    slippage_cost_usd = slippage_points * point_value

    return adjusted_price, spread_cost_usd, slippage_cost_usd


def calculate_trade_pnl(
    entry_price: float,
    exit_price: float,
    direction: int,
    lots: float,
    spread_cost_entry: float,
    spread_cost_exit: float,
    slippage_cost_entry: float,
    slippage_cost_exit: float,
    symbol: str
) -> Tuple[float, float]:
    """
    Calculate gross and net PnL for a trade.

    Returns:
        (gross_pnl, net_pnl)
    """
    config = get_symbol_config(symbol)
    point_value = config["point_value"]

    # Gross PnL (before costs)
    price_diff = exit_price - entry_price

    # Convert to points and then to USD
    if symbol.upper() in ["XAUUSD", "XAGUSD"]:
        # For metals: price is in USD, need to convert to points
        # 1 USD = (1 / point_value) points
        price_diff_points = price_diff / point_value
        gross_pnl = price_diff_points * point_value * direction * lots
    else:
        # For forex: price diff is already in pips
        gross_pnl = price_diff * point_value * direction * lots

    # Total costs
    total_spread_cost = (spread_cost_entry + spread_cost_exit) * lots
    total_slippage_cost = (slippage_cost_entry + slippage_cost_exit) * lots
    commission = COMMISSION_PER_LOT * lots

    total_costs = total_spread_cost + total_slippage_cost + commission

    net_pnl = gross_pnl - total_costs

    return gross_pnl, net_pnl


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class PropFirmBacktester:
    """
    Production-ready backtester for prop-firm challenges.

    Simulates bar-by-bar execution with realistic costs and constraints.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        predict_function: Callable[[pd.Series], int],
        initial_balance: float = INITIAL_BALANCE,
        profit_target_pct: float = PROFIT_TARGET_PCT,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        risk_per_trade_pct: float = RISK_PER_TRADE_PCT,
        verbose: bool = True,
        seed: int = 42
    ):
        """
        Initialize backtester.

        Args:
            price_df: DataFrame with columns [symbol, open, high, low, close, volume]
                     and DateTimeIndex
            predict_function: Function that takes a row (pd.Series) and returns:
                             +1 for long, -1 for short, 0 for no position
            initial_balance: Starting account balance
            profit_target_pct: Profit target as decimal (0.06 = 6%)
            max_drawdown_pct: Max drawdown as decimal (0.04 = 4%)
            risk_per_trade_pct: Risk per trade as decimal (0.005 = 0.5%)
            verbose: Print progress
            seed: Random seed for reproducibility
        """
        # Validate data
        required_cols = ["symbol", "open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in price_df.columns]
        if missing:
            raise ValueError(f"price_df missing required columns: {missing}")

        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise ValueError("price_df must have DatetimeIndex")

        # Store configuration
        self.price_df = price_df.sort_index()
        self.predict_function = predict_function
        self.initial_balance = initial_balance
        self.profit_target_pct = profit_target_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.verbose = verbose

        # Set random seed
        np.random.seed(seed)

        # Calculate thresholds
        self.target_equity = initial_balance * (1 + profit_target_pct)
        self.min_equity = initial_balance * (1 - max_drawdown_pct)

        # Initialize state
        self.equity = initial_balance
        self.peak_equity = initial_balance
        self.max_drawdown_usd = 0.0
        self.max_drawdown_pct = 0.0

        self.position: Optional[Position] = None
        self.trades: List[ClosedTrade] = []
        self.equity_curve: List[Dict] = []

        # Entry costs stored for later
        self.entry_spread_cost = 0.0
        self.entry_slippage_cost = 0.0

        # Results
        self.passed = False
        self.failure_reason: Optional[str] = None

        if self.verbose:
            print("=" * 80)
            print("PROP-FIRM MODEL BACKTESTER")
            print("=" * 80)
            print(f"Initial Balance:   ${self.initial_balance:,.2f}")
            print(f"Profit Target:     {self.profit_target_pct*100:.1f}% (${self.initial_balance * self.profit_target_pct:,.2f})")
            print(f"Target Equity:     ${self.target_equity:,.2f}")
            print(f"Max Drawdown:      {self.max_drawdown_pct*100:.1f}% (${self.initial_balance * self.max_drawdown_pct:,.2f})")
            print(f"Hard Stop:         ${self.min_equity:,.2f}")
            print(f"Risk per Trade:    {self.risk_per_trade_pct*100:.2f}%")
            print(f"Data Bars:         {len(self.price_df):,}")
            print(f"Date Range:        {self.price_df.index[0]} to {self.price_df.index[-1]}")
            print("=" * 80)
            print()

    def _open_position(self, row: pd.Series, bar_index: int, signal: int) -> None:
        """Open a new position based on signal."""
        symbol = row["symbol"]
        price = row["close"]  # Use close for execution
        timestamp = row.name  # DatetimeIndex

        # Calculate position size
        lots = calculate_position_size(
            self.equity,
            symbol,
            STOP_DISTANCE_POINTS,
            self.risk_per_trade_pct
        )

        # Apply spread and slippage to entry
        entry_price, spread_cost, slippage_cost = apply_spread_and_slippage(
            price, symbol, signal, is_entry=True
        )

        # Store costs for later
        self.entry_spread_cost = spread_cost
        self.entry_slippage_cost = slippage_cost

        # Create position
        self.position = Position(
            symbol=symbol,
            direction=signal,
            entry_price=entry_price,
            lots=lots,
            entry_time=timestamp,
            entry_bar_index=bar_index
        )

        if self.verbose:
            direction_str = "LONG" if signal == 1 else "SHORT"
            print(f"📈 OPEN {direction_str}: {symbol} @ {entry_price:.5f} | "
                  f"{lots:.2f} lots | Equity: ${self.equity:,.2f}")

    def _close_position(self, row: pd.Series, bar_index: int) -> None:
        """Close current position and record trade."""
        if self.position is None:
            return

        symbol = row["symbol"]
        price = row["close"]
        timestamp = row.name

        # Apply spread and slippage to exit
        exit_price, exit_spread_cost, exit_slippage_cost = apply_spread_and_slippage(
            price, symbol, self.position.direction, is_entry=False
        )

        # Calculate PnL
        gross_pnl, net_pnl = calculate_trade_pnl(
            self.position.entry_price,
            exit_price,
            self.position.direction,
            self.position.lots,
            self.entry_spread_cost,
            exit_spread_cost,
            self.entry_slippage_cost,
            exit_slippage_cost,
            symbol
        )

        # Update equity
        equity_before = self.equity
        self.equity += net_pnl

        # Record trade
        trade = ClosedTrade(
            symbol=symbol,
            direction=self.position.direction,
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            lots=self.position.lots,
            gross_pnl=gross_pnl,
            spread_cost=(self.entry_spread_cost + exit_spread_cost) * self.position.lots,
            commission=COMMISSION_PER_LOT * self.position.lots,
            slippage_cost=(self.entry_slippage_cost + exit_slippage_cost) * self.position.lots,
            net_pnl=net_pnl,
            equity_before=equity_before,
            equity_after=self.equity,
            entry_bar_index=self.position.entry_bar_index,
            exit_bar_index=bar_index
        )

        self.trades.append(trade)

        # Update drawdown tracking
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        current_dd = self.peak_equity - self.equity
        current_dd_pct = (current_dd / self.peak_equity * 100) if self.peak_equity > 0 else 0.0

        if current_dd > self.max_drawdown_usd:
            self.max_drawdown_usd = current_dd
            self.max_drawdown_pct = current_dd_pct

        if self.verbose:
            status = "✅" if net_pnl > 0 else "❌"
            direction_str = "LONG" if self.position.direction == 1 else "SHORT"
            print(f"{status} CLOSE {direction_str}: {symbol} @ {exit_price:.5f} | "
                  f"PnL: ${net_pnl:+.2f} | Equity: ${self.equity:,.2f} | "
                  f"DD: ${current_dd:.2f} ({current_dd_pct:.2f}%)")

        # Clear position
        self.position = None

    def _check_challenge_status(self) -> bool:
        """
        Check if challenge passed or failed.

        Returns:
            True to continue, False to stop
        """
        # Check max drawdown violation
        if self.equity <= self.min_equity:
            self.failure_reason = (
                f"Max drawdown violated: Equity ${self.equity:,.2f} <= ${self.min_equity:,.2f}"
            )
            if self.verbose:
                print()
                print("=" * 80)
                print(f"❌ CHALLENGE FAILED: {self.failure_reason}")
                print("=" * 80)
            return False

        # Check profit target reached
        if self.equity >= self.target_equity:
            self.passed = True
            if self.verbose:
                print()
                print("=" * 80)
                print(f"✅ CHALLENGE PASSED: Equity ${self.equity:,.2f} >= ${self.target_equity:,.2f}")
                print("=" * 80)
            return False

        return True

    def run(self) -> BacktestResults:
        """
        Run the backtest bar-by-bar.

        Returns:
            BacktestResults with complete analysis
        """
        # Process each bar
        for bar_index, (timestamp, row) in enumerate(self.price_df.iterrows()):
            # Get model prediction
            signal = self.predict_function(row)

            # State machine
            if self.position is None:
                # Flat - check for entry signal
                if signal in [1, -1]:
                    self._open_position(row, bar_index, signal)

            else:
                # Have position - check for exit signal
                should_exit = False

                if self.position.direction == 1:  # Long
                    if signal in [0, -1]:
                        should_exit = True
                elif self.position.direction == -1:  # Short
                    if signal in [0, 1]:
                        should_exit = True

                if should_exit:
                    self._close_position(row, bar_index)

                    # Check challenge status after trade closes
                    if not self._check_challenge_status():
                        break

                    # Check for immediate re-entry
                    if signal in [1, -1]:
                        self._open_position(row, bar_index, signal)

            # Record equity curve (every bar)
            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": self.equity,
                "drawdown_pct": (self.peak_equity - self.equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0.0,
                "in_position": self.position is not None
            })

        # Close any open position at end
        if self.position is not None:
            last_row = self.price_df.iloc[-1]
            self._close_position(last_row, len(self.price_df) - 1)

        # Final challenge check
        if not self.passed and self.failure_reason is None:
            if self.equity >= self.target_equity:
                self.passed = True
            else:
                self.failure_reason = (
                    f"Insufficient profit: ${self.equity - self.initial_balance:+.2f} "
                    f"< ${self.initial_balance * self.profit_target_pct:+.2f}"
                )

        # Calculate statistics
        results = self._calculate_results()

        # Print summary
        if self.verbose:
            self._print_summary(results)

        return results

    def _calculate_results(self) -> BacktestResults:
        """Calculate final backtest statistics."""
        num_trades = len(self.trades)

        if num_trades == 0:
            # No trades executed
            return BacktestResults(
                passed=self.passed,
                final_equity=self.equity,
                total_pnl=0.0,
                max_drawdown_usd=0.0,
                max_drawdown_pct=0.0,
                peak_equity=self.initial_balance,
                num_trades=0,
                num_wins=0,
                num_losses=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                total_costs=0.0,
                trades=[],
                equity_curve=pd.DataFrame(self.equity_curve),
                failure_reason=self.failure_reason
            )

        # Win/loss analysis
        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]

        num_wins = len(wins)
        num_losses = len(losses)
        win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0.0

        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0.0

        total_wins = sum(t.net_pnl for t in wins)
        total_losses = abs(sum(t.net_pnl for t in losses))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

        total_pnl = self.equity - self.initial_balance
        total_costs = sum(t.spread_cost + t.commission + t.slippage_cost for t in self.trades)

        equity_curve_df = pd.DataFrame(self.equity_curve)

        return BacktestResults(
            passed=self.passed,
            final_equity=self.equity,
            total_pnl=total_pnl,
            max_drawdown_usd=self.max_drawdown_usd,
            max_drawdown_pct=self.max_drawdown_pct,
            peak_equity=self.peak_equity,
            num_trades=num_trades,
            num_wins=num_wins,
            num_losses=num_losses,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_costs=total_costs,
            trades=self.trades,
            equity_curve=equity_curve_df,
            failure_reason=self.failure_reason
        )

    def _print_summary(self, results: BacktestResults) -> None:
        """Print backtest summary."""
        print()
        print("=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        print(f"Status:              {'✅ PASSED' if results.passed else '❌ FAILED'}")
        if results.failure_reason:
            print(f"Failure Reason:      {results.failure_reason}")
        print()
        print(f"Initial Balance:     ${self.initial_balance:,.2f}")
        print(f"Final Equity:        ${results.final_equity:,.2f}")
        print(f"Total PnL:           ${results.total_pnl:+,.2f} ({results.total_pnl/self.initial_balance*100:+.2f}%)")
        print(f"Peak Equity:         ${results.peak_equity:,.2f}")
        print(f"Max Drawdown:        ${results.max_drawdown_usd:.2f} ({results.max_drawdown_pct:.2f}%)")
        print()
        print(f"Total Trades:        {results.num_trades}")
        print(f"Wins:                {results.num_wins} ({results.win_rate:.1f}%)")
        print(f"Losses:              {results.num_losses}")
        print(f"Average Win:         ${results.avg_win:+.2f}")
        print(f"Average Loss:        ${results.avg_loss:+.2f}")
        print(f"Profit Factor:       {results.profit_factor:.2f}")
        print()
        print(f"Total Costs:         ${results.total_costs:,.2f}")
        print("=" * 80)


# ============================================================================
# MODEL INTERFACE - Example Implementation
# ============================================================================

def example_predict_signal(row: pd.Series) -> int:
    """
    Example prediction function (REPLACE WITH YOUR ML MODEL).

    This is a simple moving average crossover strategy for demonstration.

    Args:
        row: Current bar with columns [symbol, open, high, low, close, volume]

    Returns:
        +1: Long signal
        -1: Short signal
         0: No position / exit
    """
    # This is a DUMMY implementation
    # In production, replace with your actual ML model prediction

    # Example: Random signals (for testing only)
    signal = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
    return signal


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def load_example_data() -> pd.DataFrame:
    """Load example price data (replace with your actual data source)."""
    # Generate synthetic data for demonstration
    np.random.seed(42)

    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')

    # Simulate XAUUSD price movement
    num_bars = len(dates)
    base_price = 2650.0
    returns = np.random.normal(0.0001, 0.01, num_bars)
    close_prices = base_price * (1 + returns).cumprod()

    # Generate OHLCV
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        high = close * (1 + abs(np.random.normal(0, 0.003)))
        low = close * (1 - abs(np.random.normal(0, 0.003)))
        open_price = close * (1 + np.random.normal(0, 0.001))
        volume = np.random.randint(1000, 10000)

        data.append({
            "symbol": "XAUUSD",
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })

    df = pd.DataFrame(data, index=dates)
    return df


def main():
    """Example usage of the backtester."""
    print("Loading price data...")
    price_df = load_example_data()

    print(f"Loaded {len(price_df):,} bars")
    print(f"Date range: {price_df.index[0]} to {price_df.index[-1]}")
    print()

    # Create backtester
    backtester = PropFirmBacktester(
        price_df=price_df,
        predict_function=example_predict_signal,
        initial_balance=25_000,
        profit_target_pct=0.06,
        max_drawdown_pct=0.04,
        risk_per_trade_pct=0.005,
        verbose=True,
        seed=42
    )

    # Run backtest
    results = backtester.run()

    # Export results
    print("\nExporting results...")

    # Export trades
    if results.trades:
        trades_df = pd.DataFrame([t.to_dict() for t in results.trades])
        trades_df.to_csv("backtest_trades.csv", index=False)
        print(f"✅ Trades saved to: backtest_trades.csv")

    # Export equity curve
    results.equity_curve.to_csv("backtest_equity_curve.csv", index=False)
    print(f"✅ Equity curve saved to: backtest_equity_curve.csv")

    # Exit code
    sys.exit(0 if results.passed else 1)


if __name__ == "__main__":
    main()
