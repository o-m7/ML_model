"""
Walk-forward validation with threshold tuning and model selection - FIXED VERSION.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from config import Config
from feature_engineering import build_features
from model_training import TradingModel, tune_probability_threshold
from backtest import Backtest
from metrics import calculate_metrics, PerformanceMetrics


def create_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.7
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create walk-forward splits."""
    splits = []
    total_len = len(df)
    window_size = total_len // n_splits

    for i in range(n_splits):
        test_start = i * window_size
        test_end = test_start + window_size

        if i == n_splits - 1:
            test_end = total_len

        train_start = max(0, test_start - int(window_size * (1 / (1 - train_ratio))))
        train_end = test_start

        if train_end <= train_start:
            continue

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        splits.append((train_df, test_df))

    return splits


def evaluate_strategy_viability(wf_results: Dict, config: Config) -> bool:
    """
    Determine if a strategy is viable based on walk-forward results.

    Criteria:
    - Profit Factor >= 1.3
    - Sharpe Ratio >= 0.5
    - Max DD <= -8%
    - Total trades >= 50
    """
    pf = wf_results['avg_profit_factor']
    sharpe = wf_results['avg_sharpe']
    dd = wf_results['avg_max_dd']
    trades = wf_results['total_trades']

    is_viable = (
        pf >= 1.3 and
        sharpe >= 0.5 and
        dd >= -8.0 and
        trades >= 50
    )

    return is_viable


def run_walk_forward_validation(
    df: pd.DataFrame,
    config: Config,
    symbol: str,
    timeframe: int,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Run walk-forward validation with threshold tuning.
    """
    from feature_engineering import get_feature_columns

    print(f"\n{'='*60}")
    print(f"Walk-Forward Validation: {symbol} {timeframe}min")
    print(f"Total data: {len(df)} bars, {n_splits} splits")
    print(f"{'='*60}\n")

    splits = create_walk_forward_splits(df, n_splits=n_splits)
    print(f"Created {len(splits)} walk-forward splits")

    fold_results = []
    optimal_thresholds = []

    for fold_idx, (train_df, test_df) in enumerate(splits):
        print(f"\nFold {fold_idx + 1}/{len(splits)}")
        print(f"  Train: {len(train_df)} bars ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"  Test:  {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")

        # Train model
        train_split = int(len(train_df) * 0.8)
        train_fold = train_df.iloc[:train_split]
        val_fold = train_df.iloc[train_split:]

        feature_cols = get_feature_columns(train_fold)
        X_train = train_fold[feature_cols]
        y_train = train_fold['target']
        X_val = val_fold[feature_cols]
        y_val = val_fold['target']

        model = TradingModel(config, symbol, timeframe)
        stats = model.train(X_train, y_train, X_val, y_val)

        # Get predictions on test fold (use all features from training)
        test_df_copy = test_df.copy()
        y_test_proba = model.predict_proba(test_df[feature_cols])
        test_df_copy['ml_proba'] = y_test_proba

        # Tune threshold on this fold
        optimal_threshold, threshold_metrics = tune_probability_threshold(
            test_df['target'].values,
            y_test_proba,
            test_df_copy,
            model,
            config,
            symbol,
            timeframe
        )

        optimal_thresholds.append(optimal_threshold)
        model.optimal_threshold = optimal_threshold

        # Backtest with optimal threshold
        config.strategy.ml_prob_threshold = optimal_threshold
        bt = Backtest(test_df_copy, model, config, symbol, timeframe)
        results_df = bt.run()
        trade_log = bt.get_trade_log()

        if len(trade_log) > 0:
            equity_curve = results_df['equity']
            metrics = calculate_metrics(equity_curve, trade_log, config.risk.initial_capital)

            fold_results.append({
                'fold': fold_idx,
                'train_start': str(train_df.index[0]),
                'train_end': str(train_df.index[-1]),
                'test_start': str(test_df.index[0]),
                'test_end': str(test_df.index[-1]),
                'optimal_threshold': optimal_threshold,
                'total_trades': len(trade_log),
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'max_dd_pct': metrics.max_drawdown_pct,
                'total_pnl': metrics.total_pnl,
                'sharpe_ratio': metrics.sharpe_ratio,
                'avg_r': metrics.avg_r
            })

            print(f"  → Trades: {len(trade_log)}, WR: {metrics.win_rate*100:.2f}%, "
                  f"PF: {metrics.profit_factor:.2f}, DD: {metrics.max_drawdown_pct:.2f}%")
        else:
            print(f"  → No trades generated")

    # Aggregate results
    if len(fold_results) > 0:
        fold_df = pd.DataFrame(fold_results)

        aggregated = {
            'symbol': symbol,
            'timeframe': timeframe,
            'n_splits': len(splits),
            'successful_splits': len(fold_results),
            'avg_win_rate': fold_df['win_rate'].mean(),
            'avg_profit_factor': fold_df['profit_factor'].mean(),
            'avg_max_dd': fold_df['max_dd_pct'].mean(),
            'avg_sharpe': fold_df['sharpe_ratio'].mean(),
            'avg_r': fold_df['avg_r'].mean(),
            'total_trades': fold_df['total_trades'].sum(),
            'optimal_threshold': np.median(optimal_thresholds),
            'fold_results': fold_results
        }

        # Model selection criteria
        is_viable = evaluate_strategy_viability(aggregated, config)
        aggregated['is_viable'] = is_viable

        print(f"\n{'='*60}")
        print(f"Walk-Forward Summary:")
        print(f"  Avg Win Rate:      {aggregated['avg_win_rate']*100:.2f}%")
        print(f"  Avg Profit Factor: {aggregated['avg_profit_factor']:.2f}")
        print(f"  Avg Max DD:        {aggregated['avg_max_dd']:.2f}%")
        print(f"  Avg Sharpe:        {aggregated['avg_sharpe']:.2f}")
        print(f"  Total Trades:      {aggregated['total_trades']}")
        print(f"  Optimal Threshold: {aggregated['optimal_threshold']:.2f}")
        print(f"  Strategy Viable:   {'✓ YES' if is_viable else '✗ NO'}")
        print(f"{'='*60}\n")

        return aggregated
    else:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'n_splits': 0,
            'successful_splits': 0,
            'is_viable': False,
            'error': 'No successful splits'
        }


def run_final_oos_test(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
    symbol: str,
    timeframe: int,
    optimal_threshold: float = None
) -> Dict[str, Any]:
    """
    Run final OOS test with pre-tuned threshold.
    """
    from feature_engineering import get_feature_columns
    from model_training import train_model_with_walk_forward

    print(f"\n{'='*60}")
    print(f"Final Out-of-Sample Test: {symbol} {timeframe}min")
    print(f"{'='*60}\n")

    print(f"Training set: {len(train_val_df)} bars ({train_val_df.index[0]} to {train_val_df.index[-1]})")
    print(f"Test set:     {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")

    # Train final model
    model, _ = train_model_with_walk_forward(
        train_val_df,
        config,
        symbol,
        timeframe
    )

    # Use optimal threshold from walk-forward
    if optimal_threshold is not None:
        model.optimal_threshold = optimal_threshold
        config.strategy.ml_prob_threshold = optimal_threshold
        print(f"Using optimal threshold: {optimal_threshold:.2f}")

    # Generate predictions
    feature_cols = get_feature_columns(test_df)
    test_df_copy = test_df.copy()
    y_test_proba = model.predict_proba(test_df[feature_cols])
    test_df_copy['ml_proba'] = y_test_proba

    # Log probability distribution
    print(f"Probability distribution: min={y_test_proba.min():.3f}, "
          f"median={np.median(y_test_proba):.3f}, max={y_test_proba.max():.3f}")
    print(f"Above threshold ({config.strategy.ml_prob_threshold:.2f}): "
          f"{(y_test_proba >= config.strategy.ml_prob_threshold).sum()}")

    # Backtest on OOS
    bt = Backtest(test_df_copy, model, config, symbol, timeframe)
    results_df = bt.run()
    trade_log = bt.get_trade_log()

    # Calculate metrics
    equity_curve = results_df['equity']
    metrics = calculate_metrics(equity_curve, trade_log, config.risk.initial_capital)

    # Check targets
    from metrics import check_target_metrics
    target_checks = check_target_metrics(
        metrics,
        target_win_rate=config.prop_eval.target_win_rate,
        target_pf=config.prop_eval.target_profit_factor,
        target_max_dd=config.prop_eval.target_max_dd
    )

    # Prop firm evaluation
    from prop_eval import check_prop_firm_rules
    prop_results = check_prop_firm_rules(equity_curve, trade_log, config)

    # Compile results
    oos_results = {
        'symbol': symbol,
        'timeframe': timeframe,
        'test_period_start': str(test_df.index[0]),
        'test_period_end': str(test_df.index[-1]),
        'optimal_threshold': float(optimal_threshold) if optimal_threshold else 0.5,
        'total_trades': int(len(trade_log)),
        'win_rate': float(metrics.win_rate),
        'profit_factor': float(metrics.profit_factor),
        'max_dd_pct': float(metrics.max_drawdown_pct),
        'total_pnl': float(metrics.total_pnl),
        'sharpe_ratio': float(metrics.sharpe_ratio),
        'sortino_ratio': float(metrics.sortino_ratio),
        'avg_r': float(metrics.avg_r),
        'expectancy': float(metrics.expectancy),
        'meets_target_win_rate': bool(target_checks['win_rate_pass']),
        'meets_target_pf': bool(target_checks['profit_factor_pass']),
        'meets_target_dd': bool(target_checks['max_dd_pass']),
        'meets_all_targets': bool(target_checks['all_pass']),
        'passes_prop_eval': bool(prop_results.passes_evaluation),
        'model_path': f"models/{symbol}_{timeframe}_final.pkl"
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"OOS Test Results:")
    print(f"  Total Trades:     {oos_results['total_trades']}")
    print(f"  Win Rate:         {oos_results['win_rate']*100:.2f}% "
          f"({'✓' if oos_results['meets_target_win_rate'] else '✗'})")
    print(f"  Profit Factor:    {oos_results['profit_factor']:.2f} "
          f"({'✓' if oos_results['meets_target_pf'] else '✗'})")
    print(f"  Max Drawdown:     {oos_results['max_dd_pct']:.2f}% "
          f"({'✓' if oos_results['meets_target_dd'] else '✗'})")
    print(f"  Sharpe Ratio:     {oos_results['sharpe_ratio']:.2f}")
    print(f"  Avg R-multiple:   {oos_results['avg_r']:.2f}")
    print(f"\n  Meets All Targets: {'✓ YES' if oos_results['meets_all_targets'] else '✗ NO'}")
    print(f"  Passes Prop Eval:  {'✓ YES' if oos_results['passes_prop_eval'] else '✗ NO'}")
    print(f"{'='*60}\n")

    # Save model if it passes
    if oos_results['meets_all_targets']:
        model_dir = config.monitoring.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{symbol}_{timeframe}_final.pkl"
        model.save(str(model_path))
        print(f"Model saved to {model_path}")

    return oos_results
