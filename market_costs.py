#!/usr/bin/env python3
"""
UNIFIED MARKET COSTS MODULE
============================

Single source of truth for:
- Spreads
- Commissions
- Slippage
- TP/SL parameters

Used by BOTH backtest and live execution to ensure parity.

CRITICAL: Any changes here apply to ALL systems!
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SymbolCosts:
    """Market costs for a specific symbol."""
    spread_pips: float      # Spread in pips (XAUUSD: 1 pip = $0.10)
    commission_pct: float   # Commission as % of notional
    slippage_pct: float     # Slippage as % of notional
    min_distance_pips: float  # Min distance for TP/SL from entry (broker rule)

    def get_spread_cost(self, price: float, pip_value: float = 0.01) -> float:
        """Calculate spread cost in price units."""
        return self.spread_pips * pip_value

    def get_commission(self, notional: float) -> float:
        """Calculate commission in dollars."""
        return notional * self.commission_pct

    def get_slippage(self, notional: float) -> float:
        """Calculate expected slippage in dollars."""
        return notional * self.slippage_pct


@dataclass
class SymbolTPSL:
    """TP/SL parameters for a symbol/timeframe."""
    tp_atr_mult: float  # Take profit as ATR multiple
    sl_atr_mult: float  # Stop loss as ATR multiple

    @property
    def risk_reward_ratio(self) -> float:
        """R:R ratio."""
        return self.tp_atr_mult / self.sl_atr_mult


# ============================================================================
# MARKET COSTS - SINGLE SOURCE OF TRUTH
# ============================================================================

# Realistic costs based on typical forex/metals brokers
# Updated based on actual live execution data
MARKET_COSTS: Dict[str, SymbolCosts] = {
    'XAUUSD': SymbolCosts(
        spread_pips=3.0,        # $0.30 typical spread (1 pip = $0.10)
        commission_pct=0.00002, # 0.002% = $20 per $1M
        slippage_pct=0.00001,   # 0.001% = $10 per $1M
        min_distance_pips=5.0   # Min 5 pips from entry
    ),
    'XAGUSD': SymbolCosts(
        spread_pips=2.0,        # $0.02 typical spread (1 pip = $0.01)
        commission_pct=0.00002,
        slippage_pct=0.00001,
        min_distance_pips=3.0
    ),
    'EURUSD': SymbolCosts(
        spread_pips=0.8,        # 0.8 pips typical
        commission_pct=0.00002,
        slippage_pct=0.000005,
        min_distance_pips=1.0
    ),
    'GBPUSD': SymbolCosts(
        spread_pips=1.2,        # 1.2 pips typical
        commission_pct=0.00002,
        slippage_pct=0.000005,
        min_distance_pips=1.0
    ),
    'AUDUSD': SymbolCosts(
        spread_pips=0.9,
        commission_pct=0.00002,
        slippage_pct=0.000005,
        min_distance_pips=1.0
    ),
    'NZDUSD': SymbolCosts(
        spread_pips=1.5,
        commission_pct=0.00002,
        slippage_pct=0.000005,
        min_distance_pips=1.0
    ),
    'USDJPY': SymbolCosts(
        spread_pips=0.8,
        commission_pct=0.00002,
        slippage_pct=0.000005,
        min_distance_pips=1.0
    ),
    'USDCAD': SymbolCosts(
        spread_pips=1.5,
        commission_pct=0.00002,
        slippage_pct=0.000005,
        min_distance_pips=1.0
    ),
}


# ============================================================================
# TP/SL PARAMETERS - SINGLE SOURCE OF TRUTH
# ============================================================================

# Conservative, realistic TP/SL ratios calibrated to market microstructure
# Format: {symbol: {timeframe: SymbolTPSL}}

TP_SL_PARAMS: Dict[str, Dict[str, SymbolTPSL]] = {
    'XAUUSD': {
        '5T':  SymbolTPSL(tp_atr_mult=1.2, sl_atr_mult=1.0),  # R:R = 1.2:1
        '15T': SymbolTPSL(tp_atr_mult=1.4, sl_atr_mult=1.0),  # R:R = 1.4:1
        '30T': SymbolTPSL(tp_atr_mult=1.6, sl_atr_mult=1.0),  # R:R = 1.6:1
        '1H':  SymbolTPSL(tp_atr_mult=1.8, sl_atr_mult=1.0),  # R:R = 1.8:1
        '4H':  SymbolTPSL(tp_atr_mult=2.0, sl_atr_mult=1.0),  # R:R = 2.0:1
    },
    'XAGUSD': {
        '5T':  SymbolTPSL(tp_atr_mult=1.4, sl_atr_mult=1.0),  # Optimized for high-frequency trading
        '15T': SymbolTPSL(tp_atr_mult=1.5, sl_atr_mult=1.0),  # Conservative, proven parameters
        '30T': SymbolTPSL(tp_atr_mult=2.0, sl_atr_mult=1.0),  # FIXED: Increased from 1.8 for better RR
        '1H':  SymbolTPSL(tp_atr_mult=2.2, sl_atr_mult=1.0),  # FIXED: Increased from 2.0 for better RR
        '4H':  SymbolTPSL(tp_atr_mult=2.5, sl_atr_mult=1.0),  # FIXED: Increased from 2.2 for swing trades
    },
    'EURUSD': {
        '5T':  SymbolTPSL(tp_atr_mult=1.1, sl_atr_mult=1.0),
        '15T': SymbolTPSL(tp_atr_mult=1.3, sl_atr_mult=1.0),
        '30T': SymbolTPSL(tp_atr_mult=1.5, sl_atr_mult=1.0),
        '1H':  SymbolTPSL(tp_atr_mult=1.7, sl_atr_mult=1.0),
        '4H':  SymbolTPSL(tp_atr_mult=1.9, sl_atr_mult=1.0),
    },
    'GBPUSD': {
        '5T':  SymbolTPSL(tp_atr_mult=1.2, sl_atr_mult=1.0),
        '15T': SymbolTPSL(tp_atr_mult=1.4, sl_atr_mult=1.0),
        '30T': SymbolTPSL(tp_atr_mult=1.6, sl_atr_mult=1.0),
        '1H':  SymbolTPSL(tp_atr_mult=1.8, sl_atr_mult=1.0),
        '4H':  SymbolTPSL(tp_atr_mult=2.0, sl_atr_mult=1.0),
    },
    'AUDUSD': {
        '5T':  SymbolTPSL(tp_atr_mult=1.1, sl_atr_mult=1.0),
        '15T': SymbolTPSL(tp_atr_mult=1.3, sl_atr_mult=1.0),
        '30T': SymbolTPSL(tp_atr_mult=1.5, sl_atr_mult=1.0),
        '1H':  SymbolTPSL(tp_atr_mult=1.7, sl_atr_mult=1.0),
        '4H':  SymbolTPSL(tp_atr_mult=1.9, sl_atr_mult=1.0),
    },
    'NZDUSD': {
        '5T':  SymbolTPSL(tp_atr_mult=1.1, sl_atr_mult=1.0),
        '15T': SymbolTPSL(tp_atr_mult=1.3, sl_atr_mult=1.0),
        '30T': SymbolTPSL(tp_atr_mult=1.5, sl_atr_mult=1.0),
        '1H':  SymbolTPSL(tp_atr_mult=1.7, sl_atr_mult=1.0),
        '4H':  SymbolTPSL(tp_atr_mult=1.9, sl_atr_mult=1.0),
    },
    'USDJPY': {
        '5T':  SymbolTPSL(tp_atr_mult=1.1, sl_atr_mult=1.0),
        '15T': SymbolTPSL(tp_atr_mult=1.3, sl_atr_mult=1.0),
        '30T': SymbolTPSL(tp_atr_mult=1.5, sl_atr_mult=1.0),
        '1H':  SymbolTPSL(tp_atr_mult=1.7, sl_atr_mult=1.0),
        '4H':  SymbolTPSL(tp_atr_mult=1.9, sl_atr_mult=1.0),
    },
    'USDCAD': {
        '5T':  SymbolTPSL(tp_atr_mult=1.1, sl_atr_mult=1.0),
        '15T': SymbolTPSL(tp_atr_mult=1.3, sl_atr_mult=1.0),
        '30T': SymbolTPSL(tp_atr_mult=1.5, sl_atr_mult=1.0),
        '1H':  SymbolTPSL(tp_atr_mult=1.7, sl_atr_mult=1.0),
        '4H':  SymbolTPSL(tp_atr_mult=1.9, sl_atr_mult=1.0),
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_costs(symbol: str) -> SymbolCosts:
    """Get market costs for symbol."""
    if symbol not in MARKET_COSTS:
        raise ValueError(f"Unknown symbol: {symbol}. Add to MARKET_COSTS dict.")
    return MARKET_COSTS[symbol]


def get_tp_sl(symbol: str, timeframe: str) -> SymbolTPSL:
    """Get TP/SL params for symbol/timeframe."""
    if symbol not in TP_SL_PARAMS:
        raise ValueError(f"Unknown symbol: {symbol}. Add to TP_SL_PARAMS dict.")
    if timeframe not in TP_SL_PARAMS[symbol]:
        raise ValueError(f"Unknown timeframe {timeframe} for {symbol}. Add to TP_SL_PARAMS dict.")
    return TP_SL_PARAMS[symbol][timeframe]


def calculate_tp_sl_prices(
    symbol: str,
    timeframe: str,
    entry_price: float,
    atr: float,
    direction: str = 'long'
) -> tuple[float, float]:
    """
    Calculate TP and SL prices for a trade.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe (5T, 15T, etc.)
        entry_price: Entry price
        atr: Current ATR value
        direction: 'long' or 'short'

    Returns:
        (tp_price, sl_price)
    """
    params = get_tp_sl(symbol, timeframe)

    if direction.lower() == 'long':
        tp_price = entry_price + (atr * params.tp_atr_mult)
        sl_price = entry_price - (atr * params.sl_atr_mult)
    elif direction.lower() == 'short':
        tp_price = entry_price - (atr * params.tp_atr_mult)
        sl_price = entry_price + (atr * params.sl_atr_mult)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'long' or 'short'.")

    return tp_price, sl_price


def apply_entry_costs(
    symbol: str,
    entry_price: float,
    notional: float,
    direction: str = 'long'
) -> tuple[float, float, float]:
    """
    Apply spread, commission, and slippage to entry.

    Returns:
        (adjusted_entry_price, total_commission, total_slippage)
    """
    costs = get_costs(symbol)

    # Calculate costs
    spread_cost = costs.get_spread_cost(entry_price)
    commission = costs.get_commission(notional)
    slippage = costs.get_slippage(notional)

    # Apply to entry price (spread always widens entry)
    if direction.lower() == 'long':
        adjusted_price = entry_price + spread_cost
    else:  # short
        adjusted_price = entry_price - spread_cost

    # Slippage makes entry worse
    slippage_pct = costs.slippage_pct
    if direction.lower() == 'long':
        adjusted_price += (adjusted_price * slippage_pct)
    else:
        adjusted_price -= (adjusted_price * slippage_pct)

    return adjusted_price, commission, slippage


def apply_exit_costs(
    symbol: str,
    exit_price: float,
    notional: float,
    direction: str = 'long'
) -> tuple[float, float, float]:
    """
    Apply spread, commission, and slippage to exit.

    Returns:
        (adjusted_exit_price, total_commission, total_slippage)
    """
    costs = get_costs(symbol)

    # Calculate costs
    spread_cost = costs.get_spread_cost(exit_price)
    commission = costs.get_commission(notional)
    slippage = costs.get_slippage(notional)

    # Apply to exit price (spread always worsens exit)
    if direction.lower() == 'long':
        adjusted_price = exit_price - spread_cost
    else:  # short
        adjusted_price = exit_price + spread_cost

    # Slippage makes exit worse
    slippage_pct = costs.slippage_pct
    if direction.lower() == 'long':
        adjusted_price -= (adjusted_price * slippage_pct)
    else:
        adjusted_price += (adjusted_price * slippage_pct)

    return adjusted_price, commission, slippage


# ============================================================================
# PIP VALUE MAPPING
# ============================================================================

# Pip values for position sizing
PIP_VALUES = {
    'XAUUSD': 0.01,   # 1 pip = $0.01 (but spread quoted in $0.10 increments)
    'XAGUSD': 0.001,  # 1 pip = $0.001
    'EURUSD': 0.0001, # 1 pip = 0.0001
    'GBPUSD': 0.0001,
    'AUDUSD': 0.0001,
    'NZDUSD': 0.0001,
    'USDJPY': 0.01,   # JPY pairs
    'USDCAD': 0.0001,
}


def get_pip_value(symbol: str) -> float:
    """Get pip value for symbol."""
    return PIP_VALUES.get(symbol, 0.0001)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_all_symbols():
    """Validate that all symbols have costs and TP/SL params."""
    symbols = set(MARKET_COSTS.keys())
    tp_sl_symbols = set(TP_SL_PARAMS.keys())

    if symbols != tp_sl_symbols:
        missing_costs = tp_sl_symbols - symbols
        missing_tp_sl = symbols - tp_sl_symbols

        if missing_costs:
            print(f"⚠️  Symbols missing market costs: {missing_costs}")
        if missing_tp_sl:
            print(f"⚠️  Symbols missing TP/SL params: {missing_tp_sl}")

        return False

    print(f"✅ All {len(symbols)} symbols have costs and TP/SL params")
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("MARKET COSTS MODULE - VALIDATION")
    print("="*80 + "\n")

    validate_all_symbols()

    print("\nSample cost calculations for XAUUSD:")
    print("-" * 80)

    symbol = 'XAUUSD'
    tf = '15T'
    entry = 2650.50
    atr = 8.50
    notional = 100000

    # TP/SL
    tp, sl = calculate_tp_sl_prices(symbol, tf, entry, atr, 'long')
    print(f"Entry: ${entry:.2f}")
    print(f"ATR: ${atr:.2f}")
    print(f"TP: ${tp:.2f} (+{atr * get_tp_sl(symbol, tf).tp_atr_mult:.2f})")
    print(f"SL: ${sl:.2f} (-{atr * get_tp_sl(symbol, tf).sl_atr_mult:.2f})")
    print(f"R:R Ratio: {get_tp_sl(symbol, tf).risk_reward_ratio:.2f}:1")

    # Entry costs
    print(f"\nEntry costs (${notional:,.0f} notional):")
    adj_entry, comm, slip = apply_entry_costs(symbol, entry, notional, 'long')
    print(f"Raw entry: ${entry:.2f}")
    print(f"With costs: ${adj_entry:.2f} (degradation: ${adj_entry - entry:.2f})")
    print(f"Commission: ${comm:.2f}")
    print(f"Slippage: ${slip:.2f}")

    # Exit costs
    print(f"\nExit costs at TP (${notional:,.0f} notional):")
    adj_exit, comm, slip = apply_exit_costs(symbol, tp, notional, 'long')
    print(f"Raw exit: ${tp:.2f}")
    print(f"With costs: ${adj_exit:.2f} (degradation: ${tp - adj_exit:.2f})")
    print(f"Commission: ${comm:.2f}")
    print(f"Slippage: ${slip:.2f}")

    # Net P&L
    gross_pnl = (tp - entry) * (notional / entry)
    net_pnl = (adj_exit - adj_entry) * (notional / adj_entry) - (comm * 2) - (slip * 2)
    print(f"\nP&L Impact:")
    print(f"Gross P&L: ${gross_pnl:.2f}")
    print(f"Net P&L: ${net_pnl:.2f}")
    print(f"Cost drag: ${gross_pnl - net_pnl:.2f} ({(gross_pnl - net_pnl) / gross_pnl * 100:.1f}%)")

    print("\n" + "="*80 + "\n")
