"""
EXAMPLE USAGE
=============

This demonstrates how to use the intraday trading system.
"""

# ============================================================================
# EXAMPLE 1: Train Single Symbol/Timeframe
# ============================================================================

def example_train_single():
    """Train a single model."""
    from intraday_system.cli.train import train_single_model
    
    result = train_single_model(
        symbol='XAUUSD',
        timeframe='15T',
        config_path='intraday_system/config/settings.yaml',
        output_dir='models_intraday'
    )
    
    print(f"Training result: {result}")
    return result


# ============================================================================
# EXAMPLE 2: Generate Live Signal
# ============================================================================

def example_live_prediction():
    """Generate a live trading signal."""
    import pandas as pd
    from intraday_system.live.runner import predict
    from intraday_system.io.dataset import load_symbol_data
    
    # Load latest 200 bars
    latest_bars = load_symbol_data(
        symbol='XAUUSD',
        timeframe='15T',
        data_root='feature_store'
    ).tail(200)
    
    # Generate signal
    signal = predict(
        symbol='XAUUSD',
        timeframe='15T',
        latest_bars=latest_bars,
        models_dir='models_intraday'
    )
    
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Entry: {signal['entry_ref']:.2f}")
    print(f"SL: {signal['stop_loss']:.2f}")
    print(f"TP: {signal['take_profit']:.2f}")
    
    return signal


# ============================================================================
# EXAMPLE 3: Apply Filters to Signal
# ============================================================================

def example_filter_signal():
    """Apply post-processing filters."""
    from intraday_system.live.postprocess import apply_filters
    
    # Mock signal
    signal = {
        'signal': 'BUY',
        'confidence': 0.65,
        'entry_ref': 2650.0,
        'stop_loss': 2648.0,
        'take_profit': 2653.0,
        'expected_R': 1.5
    }
    
    # Apply filters
    filtered = apply_filters(
        signal=signal,
        current_spread=0.5,
        atr=2.0,
        last_trade_bar=100,
        current_bar=110,
        cooldown_bars=5,
        spread_atr_threshold=0.5
    )
    
    if filtered['filtered']:
        print(f"Signal filtered: {filtered['filter_reasons']}")
    else:
        print(f"Signal approved: {filtered['signal']}")
    
    return filtered


# ============================================================================
# EXAMPLE 4: Check Model Registry
# ============================================================================

def example_check_registry():
    """Check trained models in registry."""
    from intraday_system.io.registry import ModelRegistry
    
    registry = ModelRegistry('models_intraday')
    
    # Get manifest
    manifest = registry.get_manifest()
    print(f"Total models: {manifest['summary']['total_models']}")
    print(f"Ready: {manifest['summary']['ready_models']}")
    
    # List ready models
    ready = registry.list_ready_models()
    for model in ready:
        print(f"  {model['symbol']} {model['timeframe']}: PF={model['oos_metrics']['profit_factor']:.2f}")
    
    return manifest


# ============================================================================
# EXAMPLE 5: Walk-Forward CV
# ============================================================================

def example_walkforward_cv():
    """Demonstrate walk-forward cross-validation."""
    import pandas as pd
    from intraday_system.evaluation.walkforward import WalkForwardCV
    
    # Create toy data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=1000, freq='1H'),
        'value': range(1000)
    })
    
    cv = WalkForwardCV(n_folds=5, embargo_bars=50, purge_bars=25)
    splits = cv.split(df)
    
    print(f"Created {len(splits)} folds:")
    fold_info = cv.get_fold_dates(df, splits)
    for info in fold_info:
        print(f"  Fold {info['fold']}: Train={info['n_train']}, Val={info['n_val']}")
    
    return splits


# ============================================================================
# EXAMPLE 6: Calculate Metrics
# ============================================================================

def example_calculate_metrics():
    """Calculate trading metrics from trades."""
    from intraday_system.evaluation.metrics import calculate_metrics, check_benchmarks
    import yaml
    
    # Mock trades
    trades = [
        {'pnl': 150, 'return_pct': 1.5, 'commission': 10, 'slippage': 5},
        {'pnl': -100, 'return_pct': -1.0, 'commission': 10, 'slippage': 5},
        {'pnl': 200, 'return_pct': 2.0, 'commission': 10, 'slippage': 5},
        {'pnl': 180, 'return_pct': 1.8, 'commission': 10, 'slippage': 5},
    ] * 50  # 200 trades
    
    metrics = calculate_metrics(trades)
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    # Check benchmarks
    with open('intraday_system/config/settings.yaml') as f:
        config = yaml.safe_load(f)
    
    passed, failures = check_benchmarks(metrics, config['benchmarks'])
    print(f"Benchmarks passed: {passed}")
    if failures:
        print(f"Failures: {failures}")
    
    return metrics


# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("\\n" + "="*80)
    print("INTRADAY SYSTEM EXAMPLES")
    print("="*80)
    
    # Uncomment to run examples
    
    # print("\\n[Example 1] Train single model")
    # example_train_single()
    
    # print("\\n[Example 2] Live prediction")
    # example_live_prediction()
    
    print("\\n[Example 3] Filter signal")
    example_filter_signal()
    
    print("\\n[Example 4] Check registry")
    # example_check_registry()
    
    print("\\n[Example 5] Walk-forward CV")
    example_walkforward_cv()
    
    print("\\n[Example 6] Calculate metrics")
    example_calculate_metrics()

