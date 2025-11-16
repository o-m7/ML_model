#!/usr/bin/env python3
"""
Run institutional system with shorter training periods for limited data.
Use this when you have < 7 months of historical data.
"""

import sys
from pathlib import Path
from institutional_ml_trading_system import (
    TradingConfig, FeatureEngineer, LabelEngineer,
    WalkForwardValidator, load_sample_data
)

def main():
    """Run with shorter training periods."""

    # Configuration
    config = TradingConfig(
        symbol="XAUUSD",
        timeframe="15T",
        lookback_bars=5,
        signal_quantile=0.85,  # Slightly relaxed (top 15% vs top 10%)
        min_trades_per_segment=15,  # Reduced from 20
        min_profit_factor=1.5,
        min_sharpe_ratio=0.5
    )

    print("\n" + "=" * 80)
    print("INSTITUTIONAL ML SYSTEM - SHORT DATA MODE")
    print("=" * 80)
    print("\n⚙️  Configuration (adjusted for limited data):")
    print(f"   Symbol: {config.symbol}")
    print(f"   Timeframe: {config.timeframe}")
    print(f"   Lookback: {config.lookback_bars} bars")
    print(f"   Signal quantile: {config.signal_quantile} (top {(1-config.signal_quantile)*100:.0f}%)")
    print(f"   Min viable PF: {config.min_profit_factor}")
    print(f"   Min viable Sharpe: {config.min_sharpe_ratio}")
    print(f"   Min trades/segment: {config.min_trades_per_segment}")

    # Load data
    df_gold = load_sample_data("XAUUSD", config.timeframe)

    if df_gold is None or len(df_gold) == 0:
        print("\n❌ No data available. Cannot proceed.")
        sys.exit(1)

    # Handle timestamp in index or column
    if 'timestamp' not in df_gold.columns:
        if isinstance(df_gold.index, pd.DatetimeIndex):
            df_gold = df_gold.reset_index()
            if 'index' in df_gold.columns:
                df_gold = df_gold.rename(columns={'index': 'timestamp'})
            elif df_gold.index.name:
                df_gold = df_gold.rename(columns={df_gold.index.name: 'timestamp'})
        else:
            print("\n❌ No timestamp found in data!")
            sys.exit(1)

    df_gold['timestamp'] = pd.to_datetime(df_gold['timestamp'])
    date_range = (df_gold['timestamp'].max() - df_gold['timestamp'].min()).days
    months_available = date_range / 30

    print(f"\n📊 Data available: {len(df_gold):,} bars ({months_available:.1f} months, {date_range} days)")

    # Adjust parameters based on available data
    if months_available < 1.5:
        print("\n⚠️  VERY LIMITED DATA: Less than 1.5 months")
        print("   Using ultra-minimal config: 3 weeks train, 1 week test")
        train_months = 0.75  # 3 weeks
        test_months = 0.25   # 1 week
    elif months_available < 3:
        print("\n⚠️  LIMITED DATA: Less than 3 months")
        print("   Using minimal config: 1 month train, 2 weeks test")
        train_months = 1
        test_months = 0.5  # 2 weeks
    elif months_available < 5:
        print("\n📅 Using short config: 2 months train, 1 month test")
        train_months = 2
        test_months = 1
    else:
        print("\n📅 Using standard config: 3 months train, 1 month test")
        train_months = 3
        test_months = 1

    # Create validator with adjusted periods
    print(f"\n🔧 Creating segments: {train_months} month train, {test_months} month test")

    validator = WalkForwardValidator(config)

    # Override create_segments to use shorter periods
    import pandas as pd

    print(f"\n📅 Creating walk-forward segments...")
    print(f"   Training period: {train_months} months")
    print(f"   Test period: {test_months} months")

    # Ensure timestamp is a column (already handled above)
    df_gold = df_gold.sort_values('timestamp')
    min_date = df_gold['timestamp'].min()
    max_date = df_gold['timestamp'].max()

    segments = []
    current_date = min_date + pd.DateOffset(months=train_months)

    while current_date < max_date:
        # Training set: from start to current_date
        train_df = df_gold[df_gold['timestamp'] < current_date].copy()

        # Test set: next test_months
        test_end = current_date + pd.DateOffset(months=test_months)
        test_df = df_gold[(df_gold['timestamp'] >= current_date) & (df_gold['timestamp'] < test_end)].copy()

        # Adjust minimum sizes based on available data
        min_train = max(200, int(len(df_gold) * 0.3))  # At least 30% of data or 200 bars
        min_test = max(50, int(len(df_gold) * 0.1))    # At least 10% of data or 50 bars

        if len(train_df) > min_train and len(test_df) > min_test:
            segments.append((train_df, test_df))
            print(f"   Segment {len(segments)}: Train={len(train_df)}, Test={len(test_df)} "
                  f"(Test: {current_date.date()} to {test_end.date()})")

        # Move to next period (expanding window)
        current_date = test_end

    print(f"\n✓ Created {len(segments)} segments")

    if len(segments) == 0:
        print("\n❌ Could not create any segments. Need more data.")
        print(f"   Minimum: {train_months + test_months:.1f} months")
        print(f"   Have: {months_available:.1f} months")
        sys.exit(1)

    # Run validation on each segment
    all_results = []

    for seg_num, (train_df, test_df) in enumerate(segments, 1):
        print(f"\n{'=' * 80}")
        print(f"SEGMENT {seg_num}/{len(segments)}")
        print(f"{'=' * 80}")

        # Feature engineering
        train_df = FeatureEngineer.create_all_features(train_df)
        test_df = FeatureEngineer.create_all_features(test_df)

        # Label engineering
        train_df = LabelEngineer.create_profit_labels(train_df, config)
        test_df = LabelEngineer.create_profit_labels(test_df, config)

        # Balance training labels
        train_df = LabelEngineer.balance_labels(train_df, method='undersample')

        # Split training for calibration
        train_size = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:train_size]
        val_split = train_df.iloc[train_size:]

        # Train ensemble
        from institutional_ml_trading_system import EnsembleModelTrainer, ThresholdOptimizer, RealisticBacktester

        trainer = EnsembleModelTrainer(config)
        trainer.train_ensemble(train_split, val_split)

        # Get predictions
        test_predictions = trainer.predict_ensemble(test_df, method='average')
        val_predictions = trainer.predict_ensemble(val_split, method='average')

        # Optimize threshold
        threshold, _ = ThresholdOptimizer.find_optimal_threshold(
            val_predictions, val_split['target'].values, config, method='quantile'
        )

        # Backtest
        backtester = RealisticBacktester(config)
        metrics = backtester.backtest(test_df, test_predictions, threshold)
        backtester.print_metrics(metrics)

        # Store results
        result = {
            'segment': seg_num,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'threshold': threshold,
            'metrics': metrics.to_dict(),
            'is_viable': metrics.is_viable(config)
        }
        all_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)

    viable_segments = sum(1 for r in all_results if r['is_viable'])
    print(f"\nViable Segments: {viable_segments}/{len(all_results)} ({viable_segments/len(all_results)*100:.0f}%)")

    print(f"\n{'Segment':<10} {'Status':<10} {'Trades':<10} {'PF':<10} {'Sharpe':<10} {'Return %':<10}")
    print("-" * 70)
    for r in all_results:
        m = r['metrics']
        status = "✅ VIABLE" if r['is_viable'] else "❌ FAIL"
        print(f"{r['segment']:<10} {status:<10} {m['total_trades']:<10} "
              f"{m['profit_factor']:<10.2f} {m['sharpe_ratio']:<10.2f} "
              f"{m['total_return_pct']*100:<10.2f}")

    if all_results:
        avg_pf = np.mean([r['metrics']['profit_factor'] for r in all_results if r['metrics']['profit_factor'] > 0])
        avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in all_results])
        total_trades = sum(r['metrics']['total_trades'] for r in all_results)

        print(f"\nOverall Statistics:")
        print(f"   Average Profit Factor: {avg_pf:.2f}")
        print(f"   Average Sharpe Ratio:  {avg_sharpe:.2f}")
        print(f"   Total Trades:          {total_trades}")

        # Decision
        viable_pct = viable_segments / len(all_results)
        print(f"\n" + "=" * 80)
        if viable_pct >= 0.6 and avg_pf >= 1.5:
            print("✅ SYSTEM READY FOR PAPER TRADING")
            print("   ≥60% segments viable, average PF ≥ 1.5")
        elif viable_pct >= 0.4:
            print("⚠️  MARGINAL PERFORMANCE")
            print("   Consider fine-tuning or collecting more data")
        else:
            print("❌ NOT READY FOR DEPLOYMENT")
            print("   < 40% segments viable")
        print("=" * 80)

    # Save results
    import json
    output_dir = Path("institutional_ml_results")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / f"wfv_results_short_{config.symbol}_{config.timeframe}.json"
    with open(results_file, 'w') as f:
        results_json = []
        for r in all_results:
            r_copy = r.copy()
            results_json.append(r_copy)
        json.dump(results_json, f, indent=2)

    print(f"\n💾 Results saved to {results_file}")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    main()
