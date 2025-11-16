"""
Main orchestration script for institutional-grade trading system.

Runs the complete pipeline:
1. Data loading
2. Feature engineering
3. Model training
4. Backtesting
5. Walk-forward validation
6. Out-of-sample testing
7. Prop firm evaluation
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple

from config import get_default_config, Config
from data_loader import load_all_data, create_sample_data
from feature_engineering import build_features
from model_training import train_all_models, TradingModel
from backtest import Backtest
from metrics import calculate_metrics, check_target_metrics
from prop_eval import check_prop_firm_rules
from walk_forward import run_walk_forward_validation, run_final_oos_test


def setup_directories(config: Config):
    """Create necessary directories."""
    directories = [
        config.data.data_dir,
        config.monitoring.model_dir,
        Path("results"),
        Path("logs"),
        Path("reports")
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def generate_sample_data(config: Config):
    """Generate sample data if not available."""
    print("Checking for data files...")

    generated = False

    for symbol in config.data.symbols:
        for timeframe in config.data.timeframes:
            file_path = config.data.data_dir / f"{symbol}_{timeframe}.csv"

            if not file_path.exists():
                print(f"  Generating sample data: {symbol} {timeframe}min")
                df = create_sample_data(
                    symbol,
                    timeframe,
                    n_bars=20000,
                    save_path=str(file_path)
                )
                generated = True

    if not generated:
        print("  All data files found")

    return generated


def run_full_pipeline(config: Config, mode: str = 'full'):
    """
    Run the complete trading system pipeline.

    Args:
        config: Configuration object
        mode: 'full' = complete pipeline,
              'quick' = single symbol/timeframe for testing,
              'validation_only' = skip training, run validation only
    """
    print("\n" + "="*60)
    print("INSTITUTIONAL-GRADE TRADING SYSTEM")
    print("XAUUSD / XAGUSD Multi-Timeframe Strategy")
    print("="*60 + "\n")

    # Setup
    setup_directories(config)

    # Generate sample data if needed
    if mode != 'validation_only':
        generate_sample_data(config)

    # Load data
    print("\n" + "="*60)
    print("STEP 1: Loading Data")
    print("="*60)

    all_data = load_all_data(config)

    if len(all_data) == 0:
        print("Error: No data loaded. Please check data directory.")
        return

    # Build features
    print("\n" + "="*60)
    print("STEP 2: Feature Engineering")
    print("="*60)

    all_features = {}

    for (symbol, tf), splits in all_data.items():
        if mode == 'quick' and (symbol != config.data.symbols[0] or tf != config.data.timeframes[0]):
            continue

        print(f"\nProcessing {symbol} {tf}min...")

        all_features[(symbol, tf)] = {
            'train': build_features(splits['train'], config),
            'val': build_features(splits['val'], config),
            'test': build_features(splits['test'], config)
        }

        print(f"  Train: {len(all_features[(symbol, tf)]['train'])} bars")
        print(f"  Val:   {len(all_features[(symbol, tf)]['val'])} bars")
        print(f"  Test:  {len(all_features[(symbol, tf)]['test'])} bars")

        # Show regime distribution
        regime_dist = all_features[(symbol, tf)]['train']['regime'].value_counts()
        print(f"  Regime distribution: {regime_dist.to_dict()}")

    # Train models (if not validation_only mode)
    if mode != 'validation_only':
        print("\n" + "="*60)
        print("STEP 3: Model Training")
        print("="*60)

        models = train_all_models(all_data, all_features, config)

        print(f"\nTrained {len(models)} models")
    else:
        print("\nSkipping model training (validation_only mode)")
        models = {}

    # Walk-forward validation
    print("\n" + "="*60)
    print("STEP 4: Walk-Forward Validation")
    print("="*60)

    wf_results_all = {}

    for (symbol, tf), feature_splits in all_features.items():
        # Combine train + val for walk-forward
        train_val_df = pd.concat([
            feature_splits['train'],
            feature_splits['val']
        ])

        wf_results = run_walk_forward_validation(
            train_val_df,
            config,
            symbol,
            tf,
            n_splits=5
        )

        wf_results_all[(symbol, tf)] = wf_results

    # Model selection and strategy filtering
    print("\n" + "="*60)
    print("STEP 4.5: Model Selection & Strategy Filtering")
    print("="*60)

    viable_strategies, rejected_strategies = filter_viable_strategies(
        wf_results_all,
        config
    )

    if len(viable_strategies) == 0:
        print("\n" + "="*60)
        print("WARNING: No viable strategies found!")
        print("All strategies rejected. Cannot proceed to OOS testing.")
        print("="*60 + "\n")

        # Save rejected strategies only
        save_results({}, {}, rejected_strategies, config)
        return

    # Final out-of-sample testing
    print("\n" + "="*60)
    print("STEP 5: Final Out-of-Sample Testing")
    print("="*60)

    oos_results_all = {}

    for (symbol, tf), wf_result in viable_strategies.items():
        train_val_df = pd.concat([
            all_features[(symbol, tf)]['train'],
            all_features[(symbol, tf)]['val']
        ])

        test_df = all_features[(symbol, tf)]['test']

        # Use optimal threshold from walk-forward
        optimal_threshold = wf_result.get('optimal_threshold', 0.5)

        oos_results = run_final_oos_test(
            train_val_df,
            test_df,
            config,
            symbol,
            tf,
            optimal_threshold
        )

        oos_results_all[(symbol, tf)] = oos_results

    # Generate summary report
    print("\n" + "="*60)
    print("STEP 6: Summary Report")
    print("="*60)

    generate_summary_report(wf_results_all, oos_results_all, config)

    # Save all results
    save_results(wf_results_all, oos_results_all, rejected_strategies, config)

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60 + "\n")


def generate_summary_report(
    wf_results: Dict,
    oos_results: Dict,
    config: Config
):
    """Generate and print summary report."""
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80 + "\n")

    # Target thresholds
    target_wr = config.prop_eval.target_win_rate
    target_pf = config.prop_eval.target_profit_factor
    target_dd = config.prop_eval.target_max_dd

    print(f"Target Metrics:")
    print(f"  Win Rate:     ≥ {target_wr*100:.0f}%")
    print(f"  Profit Factor: ≥ {target_pf:.1f}")
    print(f"  Max Drawdown: ≤ {target_dd:.0f}%")
    print(f"\n")

    # Summary table
    summary_data = []

    for (symbol, tf) in oos_results.keys():
        oos = oos_results[(symbol, tf)]
        wf = wf_results.get((symbol, tf), {})

        summary_data.append({
            'Symbol': symbol,
            'TF': f"{tf}m",
            'Trades': oos.get('total_trades', 0),
            'WR': f"{oos.get('win_rate', 0)*100:.1f}%",
            'PF': f"{oos.get('profit_factor', 0):.2f}",
            'DD': f"{oos.get('max_dd_pct', 0):.1f}%",
            'Sharpe': f"{oos.get('sharpe_ratio', 0):.2f}",
            'R': f"{oos.get('avg_r', 0):.2f}",
            'Targets': '✓' if oos.get('meets_all_targets', False) else '✗',
            'Prop': '✓' if oos.get('passes_prop_eval', False) else '✗'
        })

    summary_df = pd.DataFrame(summary_data)

    print("Out-of-Sample Test Results:")
    print(summary_df.to_string(index=False))
    print(f"\n")

    # Overall stats
    total_passing_targets = sum(1 for r in oos_results.values() if r.get('meets_all_targets', False))
    total_passing_prop = sum(1 for r in oos_results.values() if r.get('passes_prop_eval', False))

    print(f"Overall Statistics:")
    print(f"  Total Strategies Tested:       {len(oos_results)}")
    print(f"  Meeting Target Metrics:        {total_passing_targets} / {len(oos_results)}")
    print(f"  Passing Prop Firm Evaluation:  {total_passing_prop} / {len(oos_results)}")
    print(f"\n")

    # Best performers
    if len(oos_results) > 0:
        best_wr = max(oos_results.items(), key=lambda x: x[1].get('win_rate', 0))
        best_pf = max(oos_results.items(), key=lambda x: x[1].get('profit_factor', 0))
        best_sharpe = max(oos_results.items(), key=lambda x: x[1].get('sharpe_ratio', 0))

        print(f"Best Performers:")
        print(f"  Highest Win Rate:     {best_wr[0][0]} {best_wr[0][1]}m "
              f"({best_wr[1].get('win_rate', 0)*100:.1f}%)")
        print(f"  Highest Profit Factor: {best_pf[0][0]} {best_pf[0][1]}m "
              f"({best_pf[1].get('profit_factor', 0):.2f})")
        print(f"  Highest Sharpe:       {best_sharpe[0][0]} {best_sharpe[0][1]}m "
              f"({best_sharpe[1].get('sharpe_ratio', 0):.2f})")

    print("\n" + "="*80 + "\n")


def convert_to_native_types(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj


def filter_viable_strategies(
    wf_results_all: Dict,
    config: Config
) -> Tuple[Dict, Dict]:
    """
    Filter strategies to only those that are viable based on walk-forward performance.

    Also auto-reject XAGUSD if all silver strategies are unprofitable.
    """
    viable_strategies = {}
    rejected_strategies = {}

    # Check XAGUSD viability
    xagusd_strategies = {k: v for k, v in wf_results_all.items() if k[0] == 'XAGUSD'}
    xagusd_viable = any(v.get('is_viable', False) for v in xagusd_strategies.values())

    if not xagusd_viable and len(xagusd_strategies) > 0:
        print("\n" + "="*60)
        print("WARNING: All XAGUSD strategies are UNPROFITABLE")
        print("Excluding XAGUSD from OOS testing")
        print("="*60 + "\n")

    for (symbol, tf), wf_result in wf_results_all.items():
        # Skip XAGUSD if all silver strategies failed
        if symbol == 'XAGUSD' and not xagusd_viable:
            rejected_strategies[(symbol, tf)] = {
                **wf_result,
                'rejection_reason': 'All XAGUSD strategies unprofitable'
            }
            continue

        # Check viability
        if wf_result.get('is_viable', False):
            viable_strategies[(symbol, tf)] = wf_result
            pf = wf_result.get('avg_profit_factor', 0)
            trades = wf_result.get('total_trades', 0)
            sharpe = wf_result.get('avg_sharpe', 0)
            print(f"✓ {symbol} {tf}min: APPROVED for OOS testing "
                  f"(PF={pf:.2f}, Trades={trades}, Sharpe={sharpe:.2f})")
        else:
            # Use rejection_reasons from evaluate_strategy_viability if available
            if 'rejection_reasons' in wf_result and wf_result['rejection_reasons']:
                reasons = wf_result['rejection_reasons']
            else:
                # Fallback: compute reasons here
                pf = wf_result.get('avg_profit_factor', 0)
                sharpe = wf_result.get('avg_sharpe', 0)
                trades = wf_result.get('total_trades', 0)
                dd = wf_result.get('avg_max_dd', 0)

                reasons = []
                if pf < config.validation.min_profit_factor:
                    reasons.append(f"PF={pf:.2f}<{config.validation.min_profit_factor}")
                if sharpe < config.validation.min_sharpe_ratio:
                    reasons.append(f"Sharpe={sharpe:.2f}<{config.validation.min_sharpe_ratio}")
                if trades < config.validation.min_total_trades:
                    reasons.append(f"Trades={trades}<{config.validation.min_total_trades}")
                if dd < -config.validation.max_drawdown_pct:
                    reasons.append(f"DD={dd:.1f}%<-{config.validation.max_drawdown_pct}%")

            rejected_strategies[(symbol, tf)] = {
                **wf_result,
                'rejection_reason': ', '.join(reasons)
            }
            print(f"✗ {symbol} {tf}min: REJECTED ({', '.join(reasons)})")

    print(f"\n{'='*60}")
    print(f"Model Selection Summary:")
    print(f"  Viable strategies:   {len(viable_strategies)}")
    print(f"  Rejected strategies: {len(rejected_strategies)}")
    print(f"{'='*60}\n")

    return viable_strategies, rejected_strategies


def save_results(
    wf_results: Dict,
    oos_results: Dict,
    rejected_strategies: Dict,
    config: Config
):
    """Save results to JSON files with proper type conversion."""
    results_dir = Path("results")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Walk-forward results
    wf_file = results_dir / f"walk_forward_{timestamp}.json"
    with open(wf_file, 'w') as f:
        wf_serializable = {
            f"{symbol}_{tf}": convert_to_native_types(results)
            for (symbol, tf), results in wf_results.items()
        }
        json.dump(wf_serializable, f, indent=2)

    print(f"Walk-forward results saved to {wf_file}")

    # OOS results
    oos_file = results_dir / f"oos_{timestamp}.json"
    with open(oos_file, 'w') as f:
        oos_serializable = {
            f"{symbol}_{tf}": convert_to_native_types(results)
            for (symbol, tf), results in oos_results.items()
        }
        json.dump(oos_serializable, f, indent=2)

    print(f"OOS results saved to {oos_file}")

    # Rejected strategies
    rejected_file = results_dir / f"rejected_{timestamp}.json"
    with open(rejected_file, 'w') as f:
        rejected_serializable = {
            f"{symbol}_{tf}": convert_to_native_types(results)
            for (symbol, tf), results in rejected_strategies.items()
        }
        json.dump(rejected_serializable, f, indent=2)

    print(f"Rejected strategies saved to {rejected_file}")

    # Summary CSV
    summary_data = []
    for (symbol, tf), oos in oos_results.items():
        summary_data.append({
            'symbol': symbol,
            'timeframe': tf,
            'total_trades': oos.get('total_trades', 0),
            'win_rate': oos.get('win_rate', 0),
            'profit_factor': oos.get('profit_factor', 0),
            'max_dd_pct': oos.get('max_dd_pct', 0),
            'sharpe_ratio': oos.get('sharpe_ratio', 0),
            'avg_r': oos.get('avg_r', 0),
            'meets_targets': oos.get('meets_all_targets', False),
            'passes_prop': oos.get('passes_prop_eval', False)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = results_dir / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"Summary saved to {summary_file}")


def main():
    """Main entry point."""
    # Get configuration
    config = get_default_config()

    # You can override config here or load from file
    # Example: config.data.symbols = ["XAUUSD"]
    #          config.data.timeframes = [5, 15]

    # Run pipeline
    # mode options: 'full', 'quick', 'validation_only'
    run_full_pipeline(config, mode='full')


if __name__ == "__main__":
    main()
