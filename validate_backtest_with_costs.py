#!/usr/bin/env python3
"""
BACKTEST VALIDATION WITH FIXED COSTS
=====================================

Runs realistic backtests using the unified market_costs.py module.
This validates that our cost model produces acceptable results.

Usage:
    python validate_backtest_with_costs.py --symbol XAUUSD --tf 15T
    python validate_backtest_with_costs.py --symbol XAGUSD --tf 15T
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from market_costs import get_costs, get_tp_sl, apply_entry_costs, apply_exit_costs


def load_model(symbol: str, timeframe: str) -> dict:
    """Load trained model (checks rentec → production → fast)."""
    # Try models_rentec first (Renaissance system - best)
    model_path = Path(f"models_rentec/{symbol}/{symbol}_{timeframe}.pkl")

    if not model_path.exists():
        # Fallback to production models
        model_path = Path(f"models_production/{symbol}/{symbol}_{timeframe}_PRODUCTION_READY.pkl")

    if not model_path.exists():
        # Fallback to fast models
        model_path = Path(f"models_fast/{symbol}/{symbol}_{timeframe}.pkl")

    if not model_path.exists():
        raise FileNotFoundError(f"No model found for {symbol} {timeframe} in models_rentec/, models_production/, or models_fast/")

    print(f"📦 Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load feature data."""
    data_path = Path(f"feature_store/{symbol}/{symbol}_{timeframe}.parquet")

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_parquet(data_path)

    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"📊 Loaded {len(df):,} bars from {data_path.name}")
    return df


def backtest_with_realistic_costs(
    df: pd.DataFrame,
    model: dict,
    symbol: str,
    timeframe: str,
    confidence_threshold: float = 0.55,
    initial_capital: float = 100000,
    risk_pct: float = 0.01,
    max_bars_in_trade: int = 50
) -> Dict:
    """
    Run backtest with realistic costs from market_costs.py.

    Key fixes:
    - Entry at NEXT bar open (not current close)
    - Realistic spread, commission, slippage
    - Unified TP/SL parameters
    """
    # Get model components
    features = model['results']['features']
    model_obj = model['model']

    # Get predictions
    X = df[features].fillna(0).values
    probabilities = model_obj.predict_proba(X)[:, 1]

    # Get costs and TP/SL params
    costs = get_costs(symbol)
    tp_sl_params = get_tp_sl(symbol, timeframe)

    print(f"\n⚙️  Trading Parameters:")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   TP/SL: {tp_sl_params.tp_atr_mult:.1f}R / {tp_sl_params.sl_atr_mult:.1f}R")
    print(f"   Spread: {costs.spread_pips} pips")
    print(f"   Commission: {costs.commission_pct*100:.4f}%")
    print(f"   Slippage: {costs.slippage_pct*100:.4f}%")
    print(f"   Confidence threshold: {confidence_threshold:.2f}")
    print(f"   Risk per trade: {risk_pct*100:.1f}%")

    # Trading simulation
    capital = initial_capital
    peak_capital = initial_capital
    trades = []

    print(f"\n🔄 Running backtest...")

    for i in range(len(df) - max_bars_in_trade - 1):
        # Check confidence
        if probabilities[i] < confidence_threshold:
            continue

        # Entry at NEXT bar open (realistic!)
        entry_idx = i + 1
        entry_bar = df.iloc[entry_idx]
        entry_price = entry_bar['open']

        # Get ATR
        atr = entry_bar.get('atr14', entry_price * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02

        # Calculate TP/SL
        tp_price = entry_price + (atr * tp_sl_params.tp_atr_mult)
        sl_price = entry_price - (atr * tp_sl_params.sl_atr_mult)

        # Position sizing
        risk_amount = capital * risk_pct
        sl_distance = abs(entry_price - sl_price)

        if sl_distance == 0:
            continue

        position_size = risk_amount / sl_distance
        notional = position_size * entry_price

        # Apply entry costs
        adjusted_entry, entry_comm, entry_slip = apply_entry_costs(
            symbol, entry_price, notional, 'long'
        )

        # Walk forward to find exit
        exit_price = None
        exit_reason = None
        bars_held = 0

        for j in range(entry_idx, min(entry_idx + max_bars_in_trade, len(df))):
            bar = df.iloc[j]
            bars_held = j - entry_idx

            # Check SL first (pessimistic)
            if bar['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'SL'
                break

            # Check TP
            if bar['high'] >= tp_price:
                exit_price = tp_price
                exit_reason = 'TP'
                break

        # Timeout
        if exit_price is None:
            exit_idx = min(entry_idx + max_bars_in_trade, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            exit_reason = 'timeout'

        # Apply exit costs
        adjusted_exit, exit_comm, exit_slip = apply_exit_costs(
            symbol, exit_price, notional, 'long'
        )

        # Calculate P&L
        gross_pnl = (exit_price - entry_price) * position_size
        total_costs = entry_comm + entry_slip + exit_comm + exit_slip
        net_pnl = (adjusted_exit - adjusted_entry) * position_size - total_costs

        # Update capital
        capital += net_pnl
        if capital > peak_capital:
            peak_capital = capital

        # Record trade
        trades.append({
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'adjusted_entry': adjusted_entry,
            'exit_price': exit_price,
            'adjusted_exit': adjusted_exit,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'costs': total_costs,
            'capital': capital,
            'confidence': probabilities[i]
        })

    # Calculate metrics
    if len(trades) == 0:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_r': 0,
            'sharpe_per_trade': 0,
            'max_dd_pct': 0,
            'total_return_pct': 0
        }

    trades_df = pd.DataFrame(trades)

    # Basic metrics
    wins = trades_df[trades_df['net_pnl'] > 0]
    losses = trades_df[trades_df['net_pnl'] <= 0]

    win_rate = len(wins) / len(trades_df) * 100

    total_wins = wins['net_pnl'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # R-multiple
    risk_per_trade = initial_capital * risk_pct
    avg_r = trades_df['net_pnl'].mean() / risk_per_trade

    # Drawdown
    equity_curve = trades_df['capital'].values
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / running_max
    max_dd_pct = drawdown.max() * 100

    # Sharpe
    returns = trades_df['net_pnl'] / initial_capital
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Exit reasons
    tp_rate = (trades_df['exit_reason'] == 'TP').sum() / len(trades_df) * 100
    sl_rate = (trades_df['exit_reason'] == 'SL').sum() / len(trades_df) * 100
    timeout_rate = (trades_df['exit_reason'] == 'timeout').sum() / len(trades_df) * 100

    # Cost analysis
    total_costs = trades_df['costs'].sum()
    avg_cost_per_trade = trades_df['costs'].mean()
    cost_as_pct_of_gross = total_costs / trades_df['gross_pnl'].sum() * 100 if trades_df['gross_pnl'].sum() > 0 else 0

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_r': avg_r,
        'sharpe_per_trade': sharpe,
        'max_dd_pct': max_dd_pct,
        'total_return_pct': (capital - initial_capital) / initial_capital * 100,
        'avg_bars_held': trades_df['bars_held'].mean(),
        'tp_hit_rate': tp_rate,
        'sl_hit_rate': sl_rate,
        'timeout_rate': timeout_rate,
        'total_costs': total_costs,
        'avg_cost_per_trade': avg_cost_per_trade,
        'cost_drag_pct': cost_as_pct_of_gross,
        'final_capital': capital
    }


def print_results(results: Dict):
    """Print backtest results."""
    print(f"\n{'='*80}")
    print(f"📊 BACKTEST RESULTS - {results['symbol']} {results['timeframe']}")
    print(f"{'='*80}\n")

    # Core metrics
    print(f"Total Trades:     {results['total_trades']}")
    print(f"Win Rate:         {results['win_rate']:.1f}%")
    print(f"Profit Factor:    {results['profit_factor']:.2f}")
    print(f"Avg R-multiple:   {results['avg_r']:+.3f}R")
    print(f"Sharpe/Trade:     {results['sharpe_per_trade']:.2f}")
    print(f"Max Drawdown:     {results['max_dd_pct']:.2f}%")
    print(f"Total Return:     {results['total_return_pct']:+.1f}%")

    # Exit analysis
    print(f"\nExit Reasons:")
    print(f"  TP hits:        {results['tp_hit_rate']:.1f}%")
    print(f"  SL hits:        {results['sl_hit_rate']:.1f}%")
    print(f"  Timeouts:       {results['timeout_rate']:.1f}%")

    # Cost analysis
    print(f"\nCost Analysis:")
    print(f"  Total costs:    ${results['total_costs']:.2f}")
    print(f"  Avg per trade:  ${results['avg_cost_per_trade']:.2f}")
    print(f"  Cost drag:      {results['cost_drag_pct']:.1f}% of gross P&L")

    # Trade duration
    print(f"\nTrade Duration:")
    print(f"  Avg bars held:  {results['avg_bars_held']:.1f}")

    # Pass/Fail
    print(f"\n{'='*80}")
    print(f"BENCHMARK VALIDATION:")
    print(f"{'='*80}")

    checks = [
        ('Win Rate ≥ 50%', results['win_rate'] >= 50.0),
        ('Profit Factor ≥ 1.3', results['profit_factor'] >= 1.3),
        ('Avg R ≥ 0.20', results['avg_r'] >= 0.20),
        ('Sharpe ≥ 0.20', results['sharpe_per_trade'] >= 0.20),
        ('Max DD ≤ 6%', results['max_dd_pct'] <= 6.0),
        ('Min Trades ≥ 50', results['total_trades'] >= 50),
    ]

    passed = 0
    for check_name, check_pass in checks:
        status = "✅ PASS" if check_pass else "❌ FAIL"
        print(f"{status:10s} {check_name}")
        if check_pass:
            passed += 1

    print(f"\n{'='*80}")
    if passed == len(checks):
        print(f"🎉 ALL BENCHMARKS PASSED - Model ready for deployment")
    elif passed >= len(checks) - 2:
        print(f"⚠️  MOSTLY PASSING ({passed}/{len(checks)}) - Review failures")
    else:
        print(f"❌ FAILING ({passed}/{len(checks)}) - Model needs improvement")
    print(f"{'='*80}\n")

    return passed == len(checks)


def main():
    parser = argparse.ArgumentParser(description='Validate backtest with realistic costs')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol (XAUUSD, XAGUSD, etc.)')
    parser.add_argument('--tf', type=str, required=True, help='Timeframe (5T, 15T, 30T, 1H, 4H)')
    parser.add_argument('--confidence', type=float, default=0.55, help='Confidence threshold')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"BACKTEST VALIDATION - {args.symbol} {args.tf}")
    print(f"{'='*80}\n")

    try:
        # Load model
        model = load_model(args.symbol, args.tf)

        # Load data
        df = load_data(args.symbol, args.tf)

        # Run backtest
        results = backtest_with_realistic_costs(
            df=df,
            model=model,
            symbol=args.symbol,
            timeframe=args.tf,
            confidence_threshold=args.confidence
        )

        # Print results
        passed = print_results(results)

        return 0 if passed else 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
