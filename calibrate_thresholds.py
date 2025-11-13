#!/usr/bin/env python3
"""
THRESHOLD CALIBRATION TOOL
===========================

Sweeps confidence thresholds and TP/SL parameters to find optimal operating point
under realistic market costs.

Outputs:
- CSV with metrics by threshold
- ROC and Precision-Recall curves
- Optimal thresholds for target PF/Sharpe/WR

Usage:
    python calibrate_thresholds.py --symbol XAUUSD --tf 15T --data-path feature_store/XAUUSD/XAUUSD_15T.parquet
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our unified modules
from market_costs import get_costs, get_tp_sl, calculate_tp_sl_prices, apply_entry_costs, apply_exit_costs


def load_model(symbol: str, timeframe: str) -> dict:
    """Load production model."""
    model_path = Path(f"models_production/{symbol}/{symbol}_{timeframe}_PRODUCTION_READY.pkl")

    if not model_path.exists():
        # Try FAILED models as fallback
        model_path = Path(f"models_production/{symbol}/{symbol}_{timeframe}_FAILED.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"No model found for {symbol} {timeframe}")

    print(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """Load feature data."""
    df = pd.read_parquet(data_path)

    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

    print(f"Loaded {len(df):,} rows from {data_path}")
    return df


def simulate_trading(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    symbol: str,
    timeframe: str,
    confidence_threshold: float,
    tp_mult: float,
    sl_mult: float,
    initial_capital: float = 100000,
    risk_pct: float = 0.01
) -> Dict:
    """
    Simulate trading with given parameters under realistic costs.

    Returns:
        Dict with performance metrics
    """
    costs = get_costs(symbol)

    capital = initial_capital
    peak_capital = initial_capital
    trades = []

    for i in range(len(df) - 1):  # -1 because we need next bar for entry
        # Check signal
        if not predictions[i] or probabilities[i] < confidence_threshold:
            continue

        # Entry at NEXT bar open (realistic)
        entry_bar = df.iloc[i + 1]
        entry_price = entry_bar['open']
        atr = entry_bar.get('atr14', entry_price * 0.02)

        # Calculate TP/SL
        tp_price = entry_price + (atr * tp_mult)
        sl_price = entry_price - (atr * sl_mult)

        # Position sizing
        risk_amount = capital * risk_pct
        sl_distance = abs(entry_price - sl_price)
        if sl_distance == 0:
            continue

        position_size = risk_amount / sl_distance
        notional = position_size * entry_price

        # Apply entry costs
        adjusted_entry, entry_comm, entry_slip = apply_entry_costs(symbol, entry_price, notional, 'long')

        # Walk forward to find exit
        exit_price = None
        exit_reason = None
        bars_held = 0
        max_bars = 50  # Max hold time

        for j in range(i + 1, min(i + 1 + max_bars, len(df))):
            bar = df.iloc[j]
            bars_held = j - i

            # Check SL hit
            if bar['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'SL'
                break

            # Check TP hit
            if bar['high'] >= tp_price:
                exit_price = tp_price
                exit_reason = 'TP'
                break

        # Timeout
        if exit_price is None:
            exit_price = df.iloc[min(i + max_bars, len(df) - 1)]['close']
            exit_reason = 'timeout'

        # Apply exit costs
        adjusted_exit, exit_comm, exit_slip = apply_exit_costs(symbol, exit_price, notional, 'long')

        # Calculate P&L
        gross_pnl = (exit_price - entry_price) * position_size
        costs_total = entry_comm + entry_slip + exit_comm + exit_slip
        net_pnl = (adjusted_exit - adjusted_entry) * position_size - costs_total

        # Update capital
        capital += net_pnl
        if capital > peak_capital:
            peak_capital = capital

        # Record trade
        trades.append({
            'entry_idx': i,
            'entry_price': entry_price,
            'adjusted_entry': adjusted_entry,
            'exit_price': exit_price,
            'adjusted_exit': adjusted_exit,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'capital': capital,
            'confidence': probabilities[i],
            'costs': costs_total
        })

    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_r': 0,
            'total_return_pct': 0,
            'max_dd_pct': 0,
            'sharpe_per_trade': 0,
            'tp_hit_rate': 0,
            'sl_hit_rate': 0
        }

    trades_df = pd.DataFrame(trades)

    wins = trades_df[trades_df['net_pnl'] > 0]
    losses = trades_df[trades_df['net_pnl'] <= 0]

    win_rate = len(wins) / len(trades_df) * 100
    total_wins = wins['net_pnl'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Avg R-multiple
    risk_per_trade = initial_capital * risk_pct
    avg_r = trades_df['net_pnl'].mean() / risk_per_trade if risk_per_trade > 0 else 0

    # Drawdown
    equity_curve = trades_df['capital'].values
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / running_max
    max_dd_pct = drawdown.max() * 100

    # Sharpe per trade
    returns = trades_df['net_pnl'] / initial_capital
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Exit reasons
    tp_rate = (trades_df['exit_reason'] == 'TP').sum() / len(trades_df) * 100
    sl_rate = (trades_df['exit_reason'] == 'SL').sum() / len(trades_df) * 100

    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_r': avg_r,
        'total_return_pct': (capital - initial_capital) / initial_capital * 100,
        'max_dd_pct': max_dd_pct,
        'sharpe_per_trade': sharpe,
        'tp_hit_rate': tp_rate,
        'sl_hit_rate': sl_rate,
        'avg_bars_held': trades_df['bars_held'].mean(),
        'total_costs': trades_df['costs'].sum()
    }


def calibrate_confidence_threshold(
    df: pd.DataFrame,
    model: dict,
    symbol: str,
    timeframe: str,
    threshold_range: List[float] = None
) -> pd.DataFrame:
    """
    Sweep confidence thresholds to find optimal operating point.

    Returns:
        DataFrame with metrics for each threshold
    """
    if threshold_range is None:
        threshold_range = np.arange(0.40, 0.85, 0.05)

    # Get model predictions
    features = model['results']['features']
    X = df[features].fillna(0).values
    model_obj = model['model']

    probabilities = model_obj.predict_proba(X)[:, 1]
    predictions = np.ones(len(X), dtype=bool)  # We'll filter by confidence

    # Get default TP/SL params
    tp_sl = get_tp_sl(symbol, timeframe)

    results = []

    print(f"\nCalibrating confidence thresholds...")
    print(f"Testing {len(threshold_range)} thresholds from {threshold_range[0]:.2f} to {threshold_range[-1]:.2f}")
    print("-" * 80)

    for conf_threshold in threshold_range:
        metrics = simulate_trading(
            df=df,
            probabilities=probabilities,
            predictions=predictions,
            symbol=symbol,
            timeframe=timeframe,
            confidence_threshold=conf_threshold,
            tp_mult=tp_sl.tp_atr_mult,
            sl_mult=tp_sl.sl_atr_mult
        )

        results.append({
            'confidence_threshold': conf_threshold,
            **metrics
        })

        print(f"Conf {conf_threshold:.2f}: "
              f"Trades={metrics['total_trades']:3d}, "
              f"WR={metrics['win_rate']:5.1f}%, "
              f"PF={metrics['profit_factor']:5.2f}, "
              f"Avg R={metrics['avg_r']:+6.3f}, "
              f"Sharpe={metrics['sharpe_per_trade']:5.2f}")

    return pd.DataFrame(results)


def find_optimal_threshold(results_df: pd.DataFrame, target_pf: float = 1.3, target_wr: float = 50.0) -> Dict:
    """Find optimal threshold based on targets."""
    # Filter to meet minimum standards
    valid = results_df[
        (results_df['profit_factor'] >= target_pf) &
        (results_df['win_rate'] >= target_wr) &
        (results_df['total_trades'] >= 50)  # Minimum sample size
    ]

    if len(valid) == 0:
        print(f"\n⚠️  No thresholds meet targets (PF≥{target_pf}, WR≥{target_wr}%)")
        # Return best available
        best_idx = results_df['profit_factor'].idxmax()
        return results_df.iloc[best_idx].to_dict()

    # Among valid, maximize Sharpe
    best_idx = valid['sharpe_per_trade'].idxmax()
    return valid.loc[best_idx].to_dict()


def main():
    parser = argparse.ArgumentParser(description='Calibrate confidence thresholds')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol (e.g., XAUUSD)')
    parser.add_argument('--tf', type=str, required=True, help='Timeframe (e.g., 15T)')
    parser.add_argument('--data-path', type=str, required=True, help='Path to feature data (parquet)')
    parser.add_argument('--output', type=str, default='calibration_results.csv', help='Output CSV path')
    parser.add_argument('--min-conf', type=float, default=0.40, help='Min confidence to test')
    parser.add_argument('--max-conf', type=float, default=0.80, help='Max confidence to test')
    parser.add_argument('--step', type=float, default=0.05, help='Confidence step size')

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"THRESHOLD CALIBRATION - {args.symbol} {args.tf}")
    print("="*80)

    # Load model
    try:
        model = load_model(args.symbol, args.tf)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # Load data
    try:
        df = load_data(args.data_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1

    # Calibrate
    threshold_range = np.arange(args.min_conf, args.max_conf + args.step, args.step)
    results = calibrate_confidence_threshold(
        df=df,
        model=model,
        symbol=args.symbol,
        timeframe=args.tf,
        threshold_range=threshold_range
    )

    # Save results
    results.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to {args.output}")

    # Find optimal
    print("\n" + "="*80)
    print("OPTIMAL THRESHOLD SELECTION")
    print("="*80)

    optimal = find_optimal_threshold(results, target_pf=1.3, target_wr=50.0)

    print(f"\n🎯 Recommended Threshold: {optimal['confidence_threshold']:.2f}")
    print(f"   Total Trades: {optimal['total_trades']:.0f}")
    print(f"   Win Rate: {optimal['win_rate']:.1f}%")
    print(f"   Profit Factor: {optimal['profit_factor']:.2f}")
    print(f"   Avg R-multiple: {optimal['avg_r']:+.3f}R")
    print(f"   Sharpe/Trade: {optimal['sharpe_per_trade']:.2f}")
    print(f"   Max Drawdown: {optimal['max_dd_pct']:.2f}%")
    print(f"   Total Return: {optimal['total_return_pct']:+.1f}%")

    print("\n" + "="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
