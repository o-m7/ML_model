#!/usr/bin/env python3
"""
PROP-FIRM CHALLENGE BACKTESTER
================================
Simulates a realistic funded account challenge with strict rules:
- $25,000 starting balance
- 6% profit target ($1,500 = $26,500 target)
- 4% max drawdown ($1,000 max loss = $24,000 hard stop)
- Realistic costs: spread, commission, slippage

Designed for backtesting ML trading models with prop-firm constraints.

Usage:
    python backtest_propfirm.py --input trades.csv
    python backtest_propfirm.py --input trades.csv --initial-balance 50000 --profit-target 0.08
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION - Adjustable Parameters
# ============================================================================

# Account Rules
INITIAL_BALANCE = 25_000.0      # Starting balance (USD)
PROFIT_TARGET_PCT = 0.06        # 6% profit target (= $1,500)
MAX_DRAWDOWN_PCT = 0.04         # 4% max drawdown (= $1,000 loss)

# Trading Costs
COMMISSION_PER_LOT = 7.0        # Round-turn commission per 1.0 lot (USD)

# Slippage Model (points)
SLIPPAGE_POINTS_MEAN = 0.0      # Mean slippage (points)
SLIPPAGE_POINTS_STD = 2.0       # Std dev of slippage (points)

# Symbol-Specific Configuration
# Format: {symbol: {spread_points, point_value, lot_size}}
SYMBOL_CONFIG = {
    "XAUUSD": {
        "spread_points": 20.0,   # 20 point spread (2.0 USD at current prices)
        "point_value": 0.10,     # $0.10 per point per 1.0 lot
        "contract_size": 100,    # 100 oz per lot
    },
    "XAGUSD": {
        "spread_points": 2.0,    # 2 point spread (0.02 USD)
        "point_value": 0.01,     # $0.01 per point per 1.0 lot
        "contract_size": 5000,   # 5000 oz per lot
    },
    # Forex pairs (standard)
    "EURUSD": {
        "spread_points": 1.5,
        "point_value": 10.0,     # $10 per pip per 1.0 lot
        "contract_size": 100000,
    },
    "GBPUSD": {
        "spread_points": 2.0,
        "point_value": 10.0,
        "contract_size": 100000,
    },
    "AUDUSD": {
        "spread_points": 1.5,
        "point_value": 10.0,
        "contract_size": 100000,
    },
    "NZDUSD": {
        "spread_points": 2.0,
        "point_value": 10.0,
        "contract_size": 100000,
    },
}

# Default config for unknown symbols (conservative)
DEFAULT_CONFIG = {
    "spread_points": 2.0,
    "point_value": 10.0,
    "contract_size": 100000,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Trade:
    """Represents a single completed trade."""
    timestamp: datetime
    symbol: str
    side: str               # "long" or "short"
    entry_price: float
    exit_price: float
    position_size: float    # In lots (e.g., 0.5, 1.0, 2.0)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    def __post_init__(self):
        """Validate trade data."""
        if self.side.lower() not in ["long", "short"]:
            raise ValueError(f"Invalid side: {self.side}. Must be 'long' or 'short'")
        if self.position_size <= 0:
            raise ValueError(f"Position size must be > 0, got {self.position_size}")
        if self.entry_price <= 0 or self.exit_price <= 0:
            raise ValueError(f"Prices must be > 0")

        # Normalize side
        self.side = self.side.lower()


@dataclass
class TradeResult:
    """Result of executing a trade with all costs applied."""
    trade: Trade
    gross_pnl: float        # PnL before costs (USD)
    spread_cost: float      # Spread cost (USD)
    commission: float       # Commission (USD)
    slippage_cost: float    # Slippage cost (USD)
    net_pnl: float          # Final PnL after all costs (USD)
    equity_before: float    # Account equity before this trade
    equity_after: float     # Account equity after this trade

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            "timestamp": self.trade.timestamp,
            "symbol": self.trade.symbol,
            "side": self.trade.side,
            "entry_price": self.trade.entry_price,
            "exit_price": self.trade.exit_price,
            "position_size": self.trade.position_size,
            "gross_pnl": self.gross_pnl,
            "spread_cost": self.spread_cost,
            "commission": self.commission,
            "slippage_cost": self.slippage_cost,
            "net_pnl": self.net_pnl,
            "equity_before": self.equity_before,
            "equity_after": self.equity_after,
        }


@dataclass
class BacktestResults:
    """Complete backtest results."""
    passed: bool                    # Did we pass the challenge?
    final_balance: float
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
    trades: List[TradeResult]
    equity_curve: pd.DataFrame
    failure_reason: Optional[str] = None


# ============================================================================
# COST CALCULATION FUNCTIONS
# ============================================================================

def get_symbol_config(symbol: str) -> Dict:
    """Get configuration for a symbol."""
    return SYMBOL_CONFIG.get(symbol.upper(), DEFAULT_CONFIG)


def calculate_gross_pnl(trade: Trade, config: Dict) -> float:
    """
    Calculate gross PnL (before costs) in USD.

    PnL = (exit_price - entry_price) * direction * point_value * position_size

    For XAUUSD/XAGUSD:
    - Price differences are in dollars (e.g., $2680.50 - $2675.20 = $5.30)
    - Need to convert to points, then apply point_value
    """
    price_diff = trade.exit_price - trade.entry_price

    # Direction: long = +1, short = -1
    direction = 1 if trade.side == "long" else -1

    # For metals, prices are already in USD per oz
    # Price diff in USD needs to be converted to points
    # Example: XAUUSD at $2680, 1 point = $0.10, so $1 = 10 points

    # Calculate based on contract specs
    point_value = config["point_value"]

    # For XAUUSD: $5.30 price diff = 53 points at $0.10/point = $5.30 per lot
    # For forex: 0.0015 price diff = 15 pips at $10/pip = $150 per lot

    if trade.symbol.upper() in ["XAUUSD", "XAGUSD"]:
        # Metals: convert USD price diff to points
        # XAUUSD: 1 point = $0.10, so 1 USD = 10 points
        # XAGUSD: 1 point = $0.01, so 1 USD = 100 points
        points_per_usd = 1.0 / point_value
        price_diff_points = price_diff * points_per_usd
        gross_pnl = price_diff_points * point_value * direction * trade.position_size
    else:
        # Forex: price diff is already in pips
        gross_pnl = price_diff * point_value * direction * trade.position_size

    return gross_pnl


def calculate_spread_cost(trade: Trade, config: Dict) -> float:
    """
    Calculate spread cost in USD.

    Spread is paid on entry AND exit (round-turn).
    Spread cost = spread_points * point_value * position_size
    """
    spread_points = config["spread_points"]
    point_value = config["point_value"]

    spread_cost = spread_points * point_value * trade.position_size
    return spread_cost


def calculate_commission(trade: Trade) -> float:
    """
    Calculate commission in USD.

    Commission = commission_per_lot * position_size
    """
    return COMMISSION_PER_LOT * trade.position_size


def calculate_slippage_cost(trade: Trade, config: Dict) -> float:
    """
    Calculate slippage cost in USD.

    Slippage is random, applied AGAINST the trader:
    - Long: worse entry (higher) and/or worse exit (lower)
    - Short: worse entry (lower) and/or worse exit (higher)

    We sample from normal distribution and always apply as a cost.
    """
    # Sample slippage in points (can be negative, but we take abs)
    slippage_points = np.random.normal(SLIPPAGE_POINTS_MEAN, SLIPPAGE_POINTS_STD)
    slippage_points = abs(slippage_points)  # Always a cost

    point_value = config["point_value"]

    # Convert to USD
    slippage_cost = slippage_points * point_value * trade.position_size
    return slippage_cost


def execute_trade(trade: Trade, equity_before: float) -> TradeResult:
    """
    Execute a trade with all realistic costs applied.

    Returns TradeResult with detailed breakdown.
    """
    config = get_symbol_config(trade.symbol)

    # Calculate all components
    gross_pnl = calculate_gross_pnl(trade, config)
    spread_cost = calculate_spread_cost(trade, config)
    commission = calculate_commission(trade)
    slippage_cost = calculate_slippage_cost(trade, config)

    # Net PnL after all costs
    net_pnl = gross_pnl - spread_cost - commission - slippage_cost

    # Update equity
    equity_after = equity_before + net_pnl

    return TradeResult(
        trade=trade,
        gross_pnl=gross_pnl,
        spread_cost=spread_cost,
        commission=commission,
        slippage_cost=slippage_cost,
        net_pnl=net_pnl,
        equity_before=equity_before,
        equity_after=equity_after,
    )


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def load_trades_from_csv(filepath: str) -> List[Trade]:
    """
    Load trades from CSV file.

    Required columns:
    - timestamp
    - symbol
    - side (long/short)
    - entry_price
    - exit_price
    - position_size

    Optional columns:
    - stop_loss_price
    - take_profit_price
    """
    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = ["timestamp", "symbol", "side", "entry_price", "exit_price", "position_size"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert to Trade objects
    trades = []
    for _, row in df.iterrows():
        trade = Trade(
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            side=row["side"],
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            position_size=float(row["position_size"]),
            stop_loss_price=float(row["stop_loss_price"]) if "stop_loss_price" in row and pd.notna(row["stop_loss_price"]) else None,
            take_profit_price=float(row["take_profit_price"]) if "take_profit_price" in row and pd.notna(row["take_profit_price"]) else None,
        )
        trades.append(trade)

    return trades


def run_backtest(
    trades: List[Trade],
    initial_balance: float = INITIAL_BALANCE,
    profit_target_pct: float = PROFIT_TARGET_PCT,
    max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
    verbose: bool = True,
) -> BacktestResults:
    """
    Run complete prop-firm challenge backtest.

    Rules:
    1. Start with initial_balance
    2. Stop immediately if equity drops below (initial_balance * (1 - max_drawdown_pct))
    3. Pass if equity reaches (initial_balance * (1 + profit_target_pct)) without violating max DD

    Returns:
        BacktestResults with complete analysis
    """
    if verbose:
        print("=" * 80)
        print("PROP-FIRM CHALLENGE BACKTEST")
        print("=" * 80)
        print(f"Initial Balance:   ${initial_balance:,.2f}")
        print(f"Profit Target:     {profit_target_pct*100:.1f}% (${initial_balance * profit_target_pct:,.2f})")
        print(f"Target Equity:     ${initial_balance * (1 + profit_target_pct):,.2f}")
        print(f"Max Drawdown:      {max_drawdown_pct*100:.1f}% (${initial_balance * max_drawdown_pct:,.2f})")
        print(f"Hard Stop Equity:  ${initial_balance * (1 - max_drawdown_pct):,.2f}")
        print(f"Total Trades:      {len(trades)}")
        print("=" * 80)
        print()

    # Calculate thresholds
    target_equity = initial_balance * (1 + profit_target_pct)
    min_equity = initial_balance * (1 - max_drawdown_pct)

    # Initialize tracking
    equity = initial_balance
    peak_equity = initial_balance
    max_drawdown_usd = 0.0
    max_drawdown_pct = 0.0

    trade_results = []
    passed = False
    failure_reason = None

    # Execute trades one by one
    for i, trade in enumerate(trades):
        # Execute trade with costs
        result = execute_trade(trade, equity)
        trade_results.append(result)

        # Update equity
        equity = result.equity_after

        # Update peak and drawdown
        if equity > peak_equity:
            peak_equity = equity

        current_dd_usd = peak_equity - equity
        current_dd_pct = (current_dd_usd / peak_equity) * 100 if peak_equity > 0 else 0.0

        if current_dd_usd > max_drawdown_usd:
            max_drawdown_usd = current_dd_usd
            max_drawdown_pct = current_dd_pct

        # Log trade if verbose
        if verbose:
            status = "✅" if result.net_pnl > 0 else "❌"
            print(f"{status} Trade {i+1}/{len(trades)}: {trade.symbol} {trade.side.upper()} | "
                  f"PnL: ${result.net_pnl:+.2f} | Equity: ${equity:,.2f} | "
                  f"DD from peak: ${current_dd_usd:.2f} ({current_dd_pct:.2f}%)")

        # Check max drawdown violation (from initial balance)
        if equity < min_equity:
            failure_reason = f"Max drawdown violated: Equity ${equity:,.2f} < ${min_equity:,.2f}"
            if verbose:
                print()
                print("=" * 80)
                print(f"❌ CHALLENGE FAILED: {failure_reason}")
                print("=" * 80)
            break

        # Check profit target reached
        if equity >= target_equity:
            passed = True
            if verbose:
                print()
                print("=" * 80)
                print(f"✅ CHALLENGE PASSED: Equity ${equity:,.2f} >= ${target_equity:,.2f}")
                print("=" * 80)
            break

    # If we ran out of trades without passing or failing
    if not passed and failure_reason is None:
        if equity >= target_equity:
            passed = True
        else:
            failure_reason = f"Insufficient profit: ${equity - initial_balance:+.2f} < ${initial_balance * profit_target_pct:+.2f}"

    # Calculate statistics
    num_trades = len(trade_results)
    wins = [r for r in trade_results if r.net_pnl > 0]
    losses = [r for r in trade_results if r.net_pnl <= 0]

    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0.0

    avg_win = np.mean([r.net_pnl for r in wins]) if wins else 0.0
    avg_loss = np.mean([r.net_pnl for r in losses]) if losses else 0.0

    total_wins = sum(r.net_pnl for r in wins)
    total_losses = abs(sum(r.net_pnl for r in losses))
    profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

    total_pnl = equity - initial_balance
    total_costs = sum(r.spread_cost + r.commission + r.slippage_cost for r in trade_results)

    # Create equity curve
    equity_curve_data = []
    for i, result in enumerate(trade_results):
        equity_curve_data.append({
            "trade_num": i + 1,
            "timestamp": result.trade.timestamp,
            "equity": result.equity_after,
            "pnl": result.net_pnl,
            "cumulative_pnl": result.equity_after - initial_balance,
        })

    equity_curve = pd.DataFrame(equity_curve_data)

    # Print summary
    if verbose:
        print()
        print("=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        print(f"Status:              {'✅ PASSED' if passed else '❌ FAILED'}")
        if failure_reason:
            print(f"Failure Reason:      {failure_reason}")
        print()
        print(f"Initial Balance:     ${initial_balance:,.2f}")
        print(f"Final Balance:       ${equity:,.2f}")
        print(f"Total PnL:           ${total_pnl:+,.2f} ({total_pnl/initial_balance*100:+.2f}%)")
        print(f"Peak Equity:         ${peak_equity:,.2f}")
        print(f"Max Drawdown:        ${max_drawdown_usd:.2f} ({max_drawdown_pct:.2f}%)")
        print()
        print(f"Total Trades:        {num_trades}")
        print(f"Wins:                {num_wins} ({win_rate:.1f}%)")
        print(f"Losses:              {num_losses}")
        print(f"Average Win:         ${avg_win:+.2f}")
        print(f"Average Loss:        ${avg_loss:+.2f}")
        print(f"Profit Factor:       {profit_factor:.2f}")
        print()
        print(f"Total Costs:         ${total_costs:,.2f}")
        print(f"  Spread:            ${sum(r.spread_cost for r in trade_results):,.2f}")
        print(f"  Commission:        ${sum(r.commission for r in trade_results):,.2f}")
        print(f"  Slippage:          ${sum(r.slippage_cost for r in trade_results):,.2f}")
        print("=" * 80)

    return BacktestResults(
        passed=passed,
        final_balance=equity,
        total_pnl=total_pnl,
        max_drawdown_usd=max_drawdown_usd,
        max_drawdown_pct=max_drawdown_pct,
        peak_equity=peak_equity,
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        total_costs=total_costs,
        trades=trade_results,
        equity_curve=equity_curve,
        failure_reason=failure_reason,
    )


def export_results(results: BacktestResults, output_dir: str = "backtest_results"):
    """Export backtest results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export equity curve
    equity_file = output_path / f"equity_curve_{timestamp}.csv"
    results.equity_curve.to_csv(equity_file, index=False)
    print(f"\n📊 Equity curve saved to: {equity_file}")

    # Export trade-by-trade results
    trades_data = [r.to_dict() for r in results.trades]
    trades_df = pd.DataFrame(trades_data)
    trades_file = output_path / f"trades_{timestamp}.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"📋 Trade results saved to: {trades_file}")

    # Export summary
    summary_data = {
        "metric": [
            "Status",
            "Initial Balance",
            "Final Balance",
            "Total PnL",
            "PnL %",
            "Peak Equity",
            "Max Drawdown USD",
            "Max Drawdown %",
            "Total Trades",
            "Wins",
            "Losses",
            "Win Rate %",
            "Avg Win",
            "Avg Loss",
            "Profit Factor",
            "Total Costs",
        ],
        "value": [
            "PASSED" if results.passed else "FAILED",
            f"${results.final_balance - results.total_pnl:,.2f}",
            f"${results.final_balance:,.2f}",
            f"${results.total_pnl:+,.2f}",
            f"{results.total_pnl/(results.final_balance - results.total_pnl)*100:+.2f}%",
            f"${results.peak_equity:,.2f}",
            f"${results.max_drawdown_usd:.2f}",
            f"{results.max_drawdown_pct:.2f}%",
            results.num_trades,
            results.num_wins,
            results.num_losses,
            f"{results.win_rate:.1f}%",
            f"${results.avg_win:+.2f}",
            f"${results.avg_loss:+.2f}",
            f"{results.profit_factor:.2f}",
            f"${results.total_costs:,.2f}",
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"📈 Summary saved to: {summary_file}")

    if results.failure_reason:
        print(f"\n⚠️  Failure Reason: {results.failure_reason}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prop-Firm Challenge Backtester")
    parser.add_argument("--input", required=True, help="Path to trades CSV file")
    parser.add_argument("--initial-balance", type=float, default=INITIAL_BALANCE,
                       help=f"Initial balance (default: ${INITIAL_BALANCE:,.0f})")
    parser.add_argument("--profit-target", type=float, default=PROFIT_TARGET_PCT,
                       help=f"Profit target %% (default: {PROFIT_TARGET_PCT*100:.0f}%%)")
    parser.add_argument("--max-drawdown", type=float, default=MAX_DRAWDOWN_PCT,
                       help=f"Max drawdown %% (default: {MAX_DRAWDOWN_PCT*100:.0f}%%)")
    parser.add_argument("--output-dir", default="backtest_results",
                       help="Output directory for results (default: backtest_results)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load trades
    print(f"Loading trades from: {args.input}")
    trades = load_trades_from_csv(args.input)
    print(f"Loaded {len(trades)} trades")
    print()

    # Run backtest
    results = run_backtest(
        trades,
        initial_balance=args.initial_balance,
        profit_target_pct=args.profit_target,
        max_drawdown_pct=args.max_drawdown,
        verbose=not args.quiet,
    )

    # Export results
    export_results(results, args.output_dir)

    # Exit code
    sys.exit(0 if results.passed else 1)


if __name__ == "__main__":
    main()
