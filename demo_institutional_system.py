#!/usr/bin/env python3
"""
DEMONSTRATION OF INSTITUTIONAL ML TRADING SYSTEM
================================================

This script demonstrates the complete pipeline with sample data and outputs.
Shows how each component works and produces example results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the main system
from institutional_ml_trading_system import (
    TradingConfig, FeatureEngineer, LabelEngineer,
    EnsembleModelTrainer, ThresholdOptimizer,
    RealisticBacktester, WalkForwardValidator,
    PerformanceMetrics
)


def generate_realistic_sample_data(symbol: str = "XAUUSD",
                                   n_bars: int = 15000,
                                   base_price: float = 1900.0) -> pd.DataFrame:
    """
    Generate realistic gold price data for demonstration.
    Includes realistic intraday patterns, volatility clustering, and trends.
    """
    print(f"\n🎲 Generating realistic sample data for {symbol}...")

    np.random.seed(42)

    # Generate timestamps (15-minute bars over ~150 days)
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(minutes=15*i) for i in range(n_bars)]

    # Generate returns with:
    # 1. Small drift (slight uptrend)
    # 2. Volatility clustering (GARCH-like)
    # 3. Intraday patterns (higher vol during trading hours)

    returns = []
    volatility = 0.01  # Initial volatility

    for i in range(n_bars):
        # Hour of day effect (higher volatility 8am-4pm UTC)
        hour = timestamps[i].hour
        if 8 <= hour <= 16:
            vol_multiplier = 1.5
        else:
            vol_multiplier = 0.7

        # Volatility clustering (GARCH effect)
        if i > 0:
            volatility = 0.7 * volatility + 0.3 * abs(returns[-1]) + 0.0005

        # Generate return
        ret = np.random.normal(0.00005, volatility * vol_multiplier)
        returns.append(ret)

    # Generate price from returns
    prices = base_price * np.exp(np.cumsum(returns))

    # Add realistic OHLC
    opens = prices + np.random.normal(0, 0.3, n_bars)
    highs = prices + np.abs(np.random.normal(0.5, 0.5, n_bars))
    lows = prices - np.abs(np.random.normal(0.5, 0.5, n_bars))
    closes = prices

    # Volume (higher during active hours)
    base_volume = 5000
    volume = []
    for ts in timestamps:
        hour = ts.hour
        if 8 <= hour <= 16:
            vol = int(np.random.normal(base_volume * 2, base_volume * 0.5))
        else:
            vol = int(np.random.normal(base_volume, base_volume * 0.3))
        volume.append(max(vol, 100))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': np.maximum(highs, np.maximum(opens, closes)),
        'low': np.minimum(lows, np.minimum(opens, closes)),
        'close': closes,
        'volume': volume
    })

    print(f"   ✓ Generated {len(df)} bars")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def demo_feature_engineering():
    """Demonstrate feature engineering."""
    print("\n" + "=" * 80)
    print("DEMO 1: FEATURE ENGINEERING")
    print("=" * 80)

    # Generate sample data
    df = generate_realistic_sample_data(n_bars=5000)

    # Apply feature engineering
    df_features = FeatureEngineer.create_all_features(df)

    print(f"\n📊 Feature Statistics:")
    print(f"   Total features created: {len(df_features.columns)}")
    print(f"\n   Sample features:")

    feature_samples = [
        'roc_5', 'vwma_20', 'macd', 'rsi_14',
        'atr_14', 'bb_width_20', 'vol_regime',
        'obv', 'mfi', 'zscore_50', 'distance_from_vwap'
    ]

    for feat in feature_samples:
        if feat in df_features.columns:
            values = df_features[feat].dropna()

            # Handle categorical features differently
            if pd.api.types.is_categorical_dtype(values):
                value_counts = values.value_counts()
                print(f"   {feat:25s}: categorical - {dict(value_counts)}")
            else:
                print(f"   {feat:25s}: mean={values.mean():8.4f}, "
                      f"std={values.std():8.4f}, "
                      f"min={values.min():8.4f}, "
                      f"max={values.max():8.4f}")

    return df_features


def demo_label_engineering(df: pd.DataFrame):
    """Demonstrate label engineering."""
    print("\n" + "=" * 80)
    print("DEMO 2: PROFIT-ALIGNED LABEL ENGINEERING")
    print("=" * 80)

    config = TradingConfig()

    # Create profit-aligned labels
    df_labeled = LabelEngineer.create_profit_labels(df, config)

    # Show label distribution
    label_counts = df_labeled['target'].value_counts()
    print(f"\n📊 Label Distribution:")
    for label, count in label_counts.items():
        label_name = "Long (1)" if label == 1 else "Short/Neutral (0)"
        print(f"   {label_name}: {count:,} ({count/len(df_labeled)*100:.1f}%)")

    # Show forward return statistics
    long_returns = df_labeled[df_labeled['target'] == 1]['forward_return_long']
    neutral_returns = df_labeled[df_labeled['target'] == 0]['forward_return_long']

    print(f"\n📈 Forward Return Analysis:")
    print(f"   Long signals: mean={long_returns.mean()*100:.3f}%, "
          f"median={long_returns.median()*100:.3f}%")
    print(f"   Neutral/Short: mean={neutral_returns.mean()*100:.3f}%, "
          f"median={neutral_returns.median()*100:.3f}%")

    print(f"\n💰 Transaction Cost Impact:")
    avg_cost = df_labeled['total_cost_pct'].mean()
    print(f"   Average total cost: {avg_cost*100:.3f}%")
    print(f"   Moves required for profit: >{avg_cost*100:.3f}%")

    return df_labeled


def demo_model_training(df: pd.DataFrame):
    """Demonstrate model training and calibration."""
    print("\n" + "=" * 80)
    print("DEMO 3: ENSEMBLE MODEL TRAINING & CALIBRATION")
    print("=" * 80)

    config = TradingConfig()

    # Split into train/val
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]

    print(f"\n📊 Data Split:")
    print(f"   Training:   {len(train_df):,} samples")
    print(f"   Validation: {len(val_df):,} samples")

    # Balance training data
    train_df = LabelEngineer.balance_labels(train_df, method='undersample')

    # Train ensemble
    trainer = EnsembleModelTrainer(config)
    models = trainer.train_ensemble(train_df, val_df)

    # Get predictions
    val_predictions = trainer.predict_ensemble(val_df, method='average')

    print(f"\n📊 Ensemble Prediction Statistics:")
    print(f"   Min:    {val_predictions.min():.4f}")
    print(f"   25%:    {np.quantile(val_predictions, 0.25):.4f}")
    print(f"   Median: {np.median(val_predictions):.4f}")
    print(f"   75%:    {np.quantile(val_predictions, 0.75):.4f}")
    print(f"   Max:    {val_predictions.max():.4f}")

    # Show feature importance
    trainer.print_feature_importance(top_n=15)

    return trainer, val_df, val_predictions


def demo_threshold_optimization(trainer, val_df, val_predictions):
    """Demonstrate threshold optimization."""
    print("\n" + "=" * 80)
    print("DEMO 4: DYNAMIC THRESHOLD OPTIMIZATION")
    print("=" * 80)

    config = TradingConfig()

    # Try different methods
    methods = ['quantile', 'profit', 'f1']

    results = {}
    for method in methods:
        print(f"\n--- Method: {method.upper()} ---")
        threshold, metrics = ThresholdOptimizer.find_optimal_threshold(
            val_predictions, val_df['target'].values, config, method=method
        )
        results[method] = (threshold, metrics)

    print(f"\n📊 Threshold Comparison:")
    print(f"{'Method':<12} {'Threshold':>10} {'Signals':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for method, (thresh, metrics) in results.items():
        if 'precision' in metrics:
            print(f"{method:<12} {thresh:>10.4f} {metrics['num_signals']:>8} "
                  f"{metrics['precision']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1']:>10.3f}")
        else:
            print(f"{method:<12} {thresh:>10.4f} {metrics['num_signals']:>8} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    return results['quantile'][0]  # Return quantile threshold


def demo_backtesting(df: pd.DataFrame, predictions: np.ndarray, threshold: float):
    """Demonstrate realistic backtesting."""
    print("\n" + "=" * 80)
    print("DEMO 5: REALISTIC BACKTESTING")
    print("=" * 80)

    config = TradingConfig()

    backtester = RealisticBacktester(config)
    metrics = backtester.backtest(df, predictions, threshold)

    backtester.print_metrics(metrics)

    # Show sample trades
    if len(backtester.trades) > 0:
        print(f"\n📋 Sample Trades (first 10):")
        print(f"{'Entry Time':<20} {'Entry $':>10} {'Exit $':>10} {'P&L $':>10} {'P&L %':>8} {'Prob':>8}")
        print("-" * 80)

        for trade in backtester.trades[:10]:
            entry_time = trade['entry_time']
            if isinstance(entry_time, pd.Timestamp):
                entry_time_str = entry_time.strftime('%Y-%m-%d %H:%M')
            else:
                entry_time_str = str(entry_time)

            print(f"{entry_time_str:<20} "
                  f"{trade['entry_price']:>10.2f} "
                  f"{trade['exit_price']:>10.2f} "
                  f"{trade['pnl']:>10.2f} "
                  f"{trade['pnl_pct']*100:>7.2f}% "
                  f"{trade['signal_prob']:>8.3f}")

    return metrics


def demo_walk_forward_validation():
    """Demonstrate complete walk-forward validation."""
    print("\n" + "=" * 80)
    print("DEMO 6: WALK-FORWARD VALIDATION (2 SEGMENTS)")
    print("=" * 80)

    # Generate larger dataset for WFV
    df = generate_realistic_sample_data(n_bars=12000)

    config = TradingConfig(
        lookback_bars=5,
        signal_quantile=0.85,  # Slightly relaxed for demo
        min_trades_per_segment=15  # Lower for demo
    )

    # Create validator
    validator = WalkForwardValidator(config)

    # Create only 2 segments for quick demo
    print(f"\n📅 Creating 2 walk-forward segments for demonstration...")

    if 'timestamp' not in df.columns:
        df = df.reset_index()

    df = df.sort_values('timestamp')

    # Manual segment creation for demo
    segment_1_train = df.iloc[:6000].copy()
    segment_1_test = df.iloc[6000:8000].copy()

    segment_2_train = df.iloc[:8000].copy()
    segment_2_test = df.iloc[8000:10000].copy()

    segments = [
        (segment_1_train, segment_1_test),
        (segment_2_train, segment_2_test)
    ]

    all_results = []

    for seg_num, (train_df, test_df) in enumerate(segments, 1):
        print(f"\n{'=' * 80}")
        print(f"SEGMENT {seg_num}/2")
        print(f"{'=' * 80}")
        print(f"Training: {len(train_df)} bars, Testing: {len(test_df)} bars")

        # Feature engineering
        train_df = FeatureEngineer.create_all_features(train_df)
        test_df = FeatureEngineer.create_all_features(test_df)

        # Label engineering
        train_df = LabelEngineer.create_profit_labels(train_df, config)
        test_df = LabelEngineer.create_profit_labels(test_df, config)

        # Balance
        train_df = LabelEngineer.balance_labels(train_df, method='undersample')

        # Split training for calibration
        train_size = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:train_size]
        val_split = train_df.iloc[train_size:]

        # Train
        trainer = EnsembleModelTrainer(config)
        trainer.train_ensemble(train_split, val_split)

        # Predict
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
            'threshold': threshold,
            'metrics': metrics.to_dict(),
            'is_viable': metrics.is_viable(config)
        }
        all_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)

    viable_count = sum(1 for r in all_results if r['is_viable'])
    print(f"\nViable Segments: {viable_count}/2 ({viable_count/2*100:.0f}%)")

    print(f"\n{'Segment':<10} {'Status':<10} {'Trades':<10} {'PF':<10} {'Sharpe':<10} {'Return %':<10}")
    print("-" * 70)
    for r in all_results:
        m = r['metrics']
        status = "✅ VIABLE" if r['is_viable'] else "❌ FAIL"
        print(f"{r['segment']:<10} {status:<10} {m['total_trades']:<10} "
              f"{m['profit_factor']:<10.2f} {m['sharpe_ratio']:<10.2f} "
              f"{m['total_return_pct']*100:<10.2f}")

    # Overall statistics
    if all_results:
        avg_pf = np.mean([r['metrics']['profit_factor'] for r in all_results if r['metrics']['profit_factor'] > 0])
        avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in all_results])
        total_trades = sum(r['metrics']['total_trades'] for r in all_results)

        print(f"\nOverall Statistics:")
        print(f"   Average Profit Factor: {avg_pf:.2f}")
        print(f"   Average Sharpe Ratio:  {avg_sharpe:.2f}")
        print(f"   Total Trades:          {total_trades}")

    return all_results


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("INSTITUTIONAL ML TRADING SYSTEM - COMPLETE DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows all components of the system in action.")
    print("It uses synthetic data to illustrate the pipeline.\n")

    # Demo 1: Feature Engineering
    df_features = demo_feature_engineering()

    # Demo 2: Label Engineering
    df_labeled = demo_label_engineering(df_features)

    # Demo 3: Model Training
    trainer, val_df, val_predictions = demo_model_training(df_labeled)

    # Demo 4: Threshold Optimization
    threshold = demo_threshold_optimization(trainer, val_df, val_predictions)

    # Demo 5: Backtesting
    test_size = int(len(df_labeled) * 0.15)
    test_df = df_labeled.iloc[-test_size:]
    test_predictions = trainer.predict_ensemble(test_df, method='average')
    metrics = demo_backtesting(test_df, test_predictions, threshold)

    # Demo 6: Walk-Forward Validation
    wfv_results = demo_walk_forward_validation()

    # Final summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("""
Key Takeaways:
1. ✓ Features are engineered from proven strategies (VWMA, RSI, ATR, etc.)
2. ✓ Labels account for transaction costs (profit-aligned)
3. ✓ Ensemble models (XGBoost + LightGBM + MLP) provide robust predictions
4. ✓ Probability calibration fixes squashed outputs
5. ✓ Dynamic thresholds adapt to model distribution
6. ✓ Realistic backtesting includes all costs
7. ✓ Walk-forward validation ensures out-of-sample robustness

Next Steps:
- Run the full system on real historical data
- Evaluate on complete walk-forward validation (6+ segments)
- Fine-tune hyperparameters if needed
- Deploy to paper trading for live testing
    """)

    print(f"\n💾 To run on real data:")
    print(f"   python institutional_ml_trading_system.py")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
