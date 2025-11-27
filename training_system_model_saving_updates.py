"""
═══════════════════════════════════════════════════════════════════════════════
UPDATES TO citadel_training_system_v2.py FOR MODEL SAVING
═══════════════════════════════════════════════════════════════════════════════

Add these functions to the training system and update the TrainingPipeline.run() 
method to save models and metadata for backtesting.

INSTRUCTIONS:
1. Add the save_model_for_backtest() function after the TrainingPipeline class
2. In TrainingPipeline.run(), after line "self.results = {...}", add the save call
3. Import joblib at the top: import joblib
"""

import joblib
import json
from pathlib import Path


def save_model_for_backtest(symbol: str, timeframe: str, best_model_name: str,
                            models: dict, feature_cols: list, 
                            optimal_threshold: float,
                            base_path: Path = Path("ML_model/ML_model")):
    """
    Save trained model and metadata for backtesting.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        best_model_name: Name of best model
        models: Dict of trained models
        feature_cols: List of feature column names
        optimal_threshold: Optimal confidence threshold
        base_path: Base path for saving
    """
    # Create models directory
    models_dir = base_path / "models" / symbol
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    model_path = models_dir / f"{symbol}_{timeframe}_best_model.pkl"
    best_model = models[best_model_name]
    
    joblib.dump(best_model, model_path)
    print(f"\n💾 Saved model to: {model_path}")
    
    # Save feature columns
    features_path = models_dir / f"{symbol}_{timeframe}_feature_cols.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"💾 Saved feature columns to: {features_path}")
    
    # Save metadata
    metadata_path = models_dir / f"{symbol}_{timeframe}_metadata.json"
    metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'best_model': best_model_name,
        'optimal_threshold': float(optimal_threshold),
        'n_features': len(feature_cols),
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {metadata_path}")


# ═══════════════════════════════════════════════════════════════════════════
# UPDATE TO TrainingPipeline.run() METHOD
# ═══════════════════════════════════════════════════════════════════════════

"""
In the TrainingPipeline.run() method, find the section where self.results is set
(around line 1100+), and ADD THIS CODE after the results dict is created:

        # Store results
        self.results = {
            'viable': True,
            'symbol': self.symbol,
            ...
        }
        
        # ===== ADD THIS SECTION =====
        # Save model for backtesting
        if self.results['viable'] and best_model_name and optimal_threshold:
            try:
                save_model_for_backtest(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    best_model_name=best_model_name,
                    models=models,
                    feature_cols=splits['feature_cols'],
                    optimal_threshold=optimal_threshold
                )
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to save model for backtest: {e}")
        # ===== END ADD =====
        
        print(f"\n{'#'*80}")
        print(f"# TRAINING COMPLETE")
        ...
"""


# ═══════════════════════════════════════════════════════════════════════════
# ALTERNATIVE: COMPLETE UPDATED TrainingPipeline.run() METHOD
# ═══════════════════════════════════════════════════════════════════════════

"""
If you prefer, here's the complete updated run() method with model saving integrated.
Replace the entire run() method in TrainingPipeline class with this:
"""

def run_updated(self):
    """Execute complete pipeline with model saving."""
    print(f"\n{'#'*80}")
    print(f"# CITADEL ML TRAINING SYSTEM V2.4")
    print(f"# Symbol: {self.symbol} | Timeframe: {self.timeframe}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    
    # Load data
    df, metadata = DataLoader.load_timeframe_data(self.symbol, self.timeframe)
    
    # Engineer features (includes liquidity sweeps)
    df = FeatureEngineer.engineer_all_features(df)
    
    # Find best labeling configuration
    best_tp, best_time_barrier = TripleBarrierLabeler.find_best_config(df, self.timeframe)
    
    # Label with best config (now uses timeframe-dependent spread)
    labels, r_pre, r_post = TripleBarrierLabeler.label(
        df, best_tp, CONFIG.SL_MULTIPLIER, best_time_barrier, self.timeframe
    )
    
    # Print labeling stats
    labeled_mask = (labels == 0) | (labels == 1)
    print(f"\n📊 FINAL LABELING STATISTICS")
    print(f"{'='*80}")
    print(f"   TP: {best_tp:.1f}x ATR")
    print(f"   Time Barrier: {best_time_barrier} bars")
    
    spread_r = CONFIG.SPREAD_R_BY_TIMEFRAME.get(self.timeframe, 0.05)
    print(f"   Spread: {spread_r:.3f}R (timeframe-dependent)")
    
    pre_pf = RiskMetrics.calculate_profit_factor(r_pre[labeled_mask].values)
    post_pf = RiskMetrics.calculate_profit_factor(r_post[labeled_mask].values)
    wr = (labels[labeled_mask] == 1).sum() / labeled_mask.sum()
    
    print(f"   Pre-cost PF: {pre_pf:.2f}")
    print(f"   Post-cost PF: {post_pf:.2f}")
    print(f"   Win Rate: {wr:.1%}")
    
    if post_pf < 1.0:
        print(f"\n   ⚠️  WARNING: Post-cost unprofitable!")
    
    # Walk-forward validation
    if self.enable_walk_forward:
        wf_results = WalkForwardValidator.run_walk_forward(
            df, labels, r_post, CONFIG.WF_N_SPLITS
        )
    
    # Chronological split
    splits = DataSplitter.split_chronological(df, labels, r_post)
    
    # Train models
    models = ModelFactory.train_all_models(
        splits['X_train'], splits['X_val'],
        splits['y_train'], splits['y_val']
    )
    
    # Evaluate with ROBUST selection
    results_raw = ModelEvaluator.evaluate_all_models(
        models, splits['X_test'], splits['y_test'], 
        splits['r_test'], self.timeframe
    )
    
    # Select best with guardrails
    best_model_name = ModelEvaluator.select_best_model(results_raw)
    
    if best_model_name is None:
        print(f"\n❌ NO VIABLE MODEL FOR {self.timeframe}")
        self.results = {'viable': False, 'timeframe': self.timeframe}
        return self.results
    
    best_model = models[best_model_name]
    
    # Optimize threshold
    val_ts = splits['val_ts']['timestamp']
    val_days = (val_ts.max() - val_ts.min()).total_seconds() / 86400
    
    optimal_threshold = ConfidenceFilter.find_optimal_threshold(
        best_model, splits['X_val'], splits['y_val'],
        splits['r_val'], self.timeframe, val_days
    )
    
    if optimal_threshold is None:
        print(f"\n❌ NO VALID THRESHOLD FOR {self.timeframe}")
        self.results = {'viable': False, 'timeframe': self.timeframe,
                      'best_model_name': best_model_name}
        return self.results
    
    # Final evaluation
    print(f"\n{'='*80}")
    print(f"POST-THRESHOLD TEST SET EVALUATION")
    print(f"{'='*80}")
    
    filtered_metrics = ModelEvaluator.evaluate_with_threshold(
        best_model, splits['X_test'], splits['y_test'],
        splits['r_test'], optimal_threshold
    )
    
    print(f"\n📊 FILTERED PERFORMANCE:")
    print(f"   Trades: {filtered_metrics['total_trades']}")
    print(f"   Win Rate: {filtered_metrics['win_rate']:.1%}")
    print(f"   Profit Factor: {filtered_metrics['profit_factor']:.2f}")
    print(f"   Sharpe: {filtered_metrics['sharpe']:.2f}")
    print(f"   Max DD: {filtered_metrics['max_drawdown_pct']:.1f}%")
    
    # Regime analysis with liquidity sweep breakdown
    test_indices = splits['test_ts'].index
    df_test = df.loc[test_indices]
    
    X_test_scaled = best_model['scaler'].transform(splits['X_test'])
    y_test_pred = best_model['model'].predict(X_test_scaled)
    
    regime_analysis = RegimeAnalyzer.analyze_by_regime(
        df_test, splits['y_test'], y_test_pred, splits['r_test']
    )
    
    # Print comparison
    ModelEvaluator.print_comparison_table(results_raw)
    
    # Store results
    self.results = {
        'viable': True,
        'symbol': self.symbol,
        'timeframe': self.timeframe,
        'best_tp_mult': best_tp,
        'time_barrier': best_time_barrier,
        'spread_r': spread_r,
        'pre_cost_pf': pre_pf,
        'post_cost_pf': post_pf,
        'optimal_threshold': optimal_threshold,
        'best_model_name': best_model_name,
        'models': models,
        'results_raw': results_raw,
        'filtered_metrics': filtered_metrics,
        'regime_analysis': regime_analysis,
        'feature_cols': splits['feature_cols']
    }
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE MODEL FOR BACKTESTING (NEW)
    # ═══════════════════════════════════════════════════════════════
    try:
        save_model_for_backtest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            best_model_name=best_model_name,
            models=models,
            feature_cols=splits['feature_cols'],
            optimal_threshold=optimal_threshold
        )
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to save model for backtest: {e}")
    
    print(f"\n{'#'*80}")
    print(f"# TRAINING COMPLETE")
    print(f"# Best Model: {best_model_name}")
    print(f"# Filtered WR: {filtered_metrics['win_rate']:.1%}")
    print(f"# Filtered PF: {filtered_metrics['profit_factor']:.2f}")
    print(f"# Max DD: {filtered_metrics['max_drawdown_pct']:.1f}%")
    print(f"# Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")
    
    return self.results