"""
Walk-forward validation and out-of-sample testing module.

Implements rolling window retraining and validation to prevent overfitting
and assess true out-of-sample performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from config import Config
from feature_engineering import build_features
from model_training import TradingModel
from backtest import Backtest
from metrics import calculate_metrics, PerformanceMetrics


def create_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.7
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward splits.

    Each split has:
    - Training set: earlier data
    - Test set: immediately following period

    Args:
        df: Full dataset
        n_splits: Number of walk-forward windows
        train_ratio: Ratio of train to total window

    Returns:
        List of (train_df, test_df) tuples
    """
    splits = []

    total_len = len(df)
    window_size = total_len // n_splits

    for i in range(n_splits):
        # Test window
        test_start = i * window_size
        test_end = test_start + window_size

        # Handle last split
        if i == n_splits - 1:
            test_end = total_len

        # Train window (all data before test window, or rolling window)
        train_start = max(0, test_start - int(window_size * (1 / (1 - train_ratio))))
        train_end = test_start

        if train_end <= train_start:
            continue

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        splits.append((train_df, test_df))

    return splits


def run_walk_forward_validation(
    df: pd.DataFrame,
    config: Config,
    symbol: str,
    timeframe: int,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Run walk-forward validation.

    For each split:
    1. Train model on training window
    2. Backtest on test window
    3. Calculate metrics

    Args:
        df: Full dataset with features
        config: Configuration
        symbol: Symbol name
        timeframe: Timeframe in minutes
        n_splits: Number of walk-forward windows

    Returns:
        Dict with aggregated results
    """
    print(f"\n{'='*60}")
    print(f"Walk-Forward Validation: {symbol} {timeframe}min")
    print(f"Total data: {len(df)} bars, {n_splits} splits")
    print(f"{'='*60}\n")

    # Create splits
    splits = create_walk_forward_splits(df, n_splits=n_splits)

    print(f"Created {len(splits)} walk-forward splits")

    fold_results = []

    for fold_idx, (train_df, test_df) in enumerate(splits):
        print(f"\nFold {fold_idx + 1}/{len(splits)}")
        print(f"  Train: {len(train_df)} bars ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"  Test:  {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")

        # Train model on this fold
        from model_training import train_model_with_walk_forward
        from feature_engineering import get_feature_columns

        # Use a subset of the train data for validation within training
        train_split = int(len(train_df) * 0.8)
        train_fold = train_df.iloc[:train_split]
        val_fold = train_df.iloc[train_split:]

        # Get feature columns
        feature_cols = get_feature_columns(train_fold)

        # Prepare training data
        X_train = train_fold[feature_cols]
        y_train = train_fold['target']
        X_val = val_fold[feature_cols]
        y_val = val_fold['target']

        model = TradingModel(config, symbol, timeframe)
        model.train(X_train, y_train, X_val, y_val)

        test_df_copy = test_df.copy()
        test_df_copy['ml_proba'] = model.predict_proba(test_df[feature_cols])

        # Backtest on test window
        bt = Backtest(test_df_copy, model, config, symbol, timeframe)
        results_df = bt.run()

        trade_log = bt.get_trade_log()

        # Calculate metrics
        if len(trade_log) > 0:
            equity_curve = results_df['equity']
            metrics = calculate_metrics(equity_curve, trade_log, config.risk.initial_capital)

            fold_results.append({
                'fold': fold_idx,
                'train_start': str(train_df.index[0]),
                'train_end': str(train_df.index[-1]),
                'test_start': str(test_df.index[0]),
                'test_end': str(test_df.index[-1]),
                'total_trades': len(trade_log),
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'max_dd_pct': metrics.max_drawdown_pct,
                'total_pnl': metrics.total_pnl,
                'sharpe_ratio': metrics.sharpe_ratio,
                'avg_r': metrics.avg_r
            })

            print(f"  Trades: {len(trade_log)}, Win Rate: {metrics.win_rate*100:.2f}%, "
                  f"PF: {metrics.profit_factor:.2f}, DD: {metrics.max_drawdown_pct:.2f}%")
        else:
            print(f"  No trades generated")

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
            'fold_results': fold_results
        }

        print(f"\n{'='*60}")
        print(f"Walk-Forward Summary:")
        print(f"  Avg Win Rate:     {aggregated['avg_win_rate']*100:.2f}%")
        print(f"  Avg Profit Factor: {aggregated['avg_profit_factor']:.2f}")
        print(f"  Avg Max DD:       {aggregated['avg_max_dd']:.2f}%")
        print(f"  Avg Sharpe:       {aggregated['avg_sharpe']:.2f}")
        print(f"  Total Trades:     {aggregated['total_trades']}")
        print(f"{'='*60}\n")

        return aggregated
    else:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'n_splits': 0,
            'successful_splits': 0,
            'error': 'No successful splits'
        }


def run_final_oos_test(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
    symbol: str,
    timeframe: int
) -> Dict[str, Any]:
    """
    Run final out-of-sample test.

    Train on train+val data, test on completely held-out test set.

    Args:
        train_val_df: Combined training and validation data
        test_df: Final test set (never seen during training)
        config: Configuration
        symbol: Symbol name
        timeframe: Timeframe in minutes

    Returns:
        Dict with final OOS results
    """
    print(f"\n{'='*60}")
    print(f"Final Out-of-Sample Test: {symbol} {timeframe}min")
    print(f"{'='*60}\n")

    print(f"Training set: {len(train_val_df)} bars ({train_val_df.index[0]} to {train_val_df.index[-1]})")
    print(f"Test set:     {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")

    # Train final model
    from model_training import train_model_with_walk_forward

    model, _ = train_model_with_walk_forward(
        train_val_df,
        config,
        symbol,
        timeframe
    )

    # Generate predictions for test set
    from feature_engineering import get_feature_columns
    feature_cols = get_feature_columns(test_df)

    test_df_copy = test_df.copy()
    test_df_copy['ml_proba'] = model.predict_proba(test_df[feature_cols])

    # Backtest on OOS test set
    bt = Backtest(test_df_copy, model, config, symbol, timeframe)
    results_df = bt.run()

    trade_log = bt.get_trade_log()

    # Calculate metrics
    equity_curve = results_df['equity']
    metrics = calculate_metrics(equity_curve, trade_log, config.risk.initial_capital)

    # Check if meets targets
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
        'total_trades': len(trade_log),
        'win_rate': metrics.win_rate,
        'profit_factor': metrics.profit_factor,
        'max_dd_pct': metrics.max_drawdown_pct,
        'total_pnl': metrics.total_pnl,
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'avg_r': metrics.avg_r,
        'expectancy': metrics.expectancy,
        'meets_target_win_rate': target_checks['win_rate_pass'],
        'meets_target_pf': target_checks['profit_factor_pass'],
        'meets_target_dd': target_checks['max_dd_pass'],
        'meets_all_targets': target_checks['all_pass'],
        'passes_prop_eval': prop_results.passes_evaluation,
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


if __name__ == "__main__":
    from config import get_default_config
    from data_loader import load_all_data, create_sample_data

    config = get_default_config()

    # Create sample data
    symbol = "XAUUSD"
    timeframe = 5

    file_path = config.data.data_dir / f"{symbol}_{timeframe}.csv"
    if not file_path.exists():
        df = create_sample_data(symbol, timeframe, n_bars=20000, save_path=str(file_path))

    # Load data
    all_data = load_all_data(config)

    if (symbol, timeframe) in all_data:
        # Build features
        print("Building features...")
        train_df = all_data[(symbol, timeframe)]['train']
        val_df = all_data[(symbol, timeframe)]['val']
        test_df = all_data[(symbol, timeframe)]['test']

        train_features = build_features(train_df, config)
        val_features = build_features(val_df, config)
        test_features = build_features(test_df, config)

        # Combine train + val for walk-forward
        train_val_features = pd.concat([train_features, val_features])

        # Run walk-forward validation
        wf_results = run_walk_forward_validation(
            train_val_features,
            config,
            symbol,
            timeframe,
            n_splits=3
        )

        # Run final OOS test
        oos_results = run_final_oos_test(
            train_val_features,
            test_features,
            config,
            symbol,
            timeframe
        )

        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        with open(results_dir / f"{symbol}_{timeframe}_wf_{timestamp}.json", 'w') as f:
            json.dump(wf_results, f, indent=2)

        with open(results_dir / f"{symbol}_{timeframe}_oos_{timestamp}.json", 'w') as f:
            json.dump(oos_results, f, indent=2)

        print(f"\nResults saved to {results_dir}")
