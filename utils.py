"""
XAUUSD Multi-Model Training Pipeline - Main Orchestrator

Production-grade training system for XAUUSD trading models.
Implements walk-forward validation, multiple model architectures,
and comprehensive performance analysis.

Author: Quantitative Engineering Team
Date: 2024
"""

import yaml
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import pipeline modules
from data_loader import DataLoader
from labeling import TripleBarrierLabeler, MetaLabeler
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from validator import WalkForwardValidator, PerformanceCalculator
from risk_manager import RiskManager
from exporter import ModelExporter


def setup_logging(log_level: str = 'INFO'):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training pipeline orchestrator."""
    
    # ========== SETUP ==========
    print("="*60)
    print("XAUUSD MODEL TRAINING PIPELINE")
    print("="*60)
    
    config = load_config()
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducibility
    np.random.seed(config['random_seed'])
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Training models: {config['models']['train']}")
    
    # ========== DATA LOADING ==========
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DATA LOADING")
    logger.info("="*60)
    
    data_loader = DataLoader(config)
    df = data_loader.load_data()
    
    logger.info(f"Loaded {len(df)} bars of {config['data']['primary_timeframe']} data")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Features: {len(df.columns)}")
    
    # ========== LABELING ==========
    logger.info("\n" + "="*60)
    logger.info("STEP 2: TRIPLE-BARRIER LABELING")
    logger.info("="*60)
    
    labeler = TripleBarrierLabeler(
        take_profit_atr=config['labeling']['take_profit_atr_multiples'][0],  # Start with first option
        stop_loss_atr=config['labeling']['stop_loss_atr_multiple'],
        time_barrier_bars=config['labeling']['time_barrier_bars'],
        min_return_threshold=config['labeling']['min_return_threshold']
    )
    
    # Optimize barrier ratios
    logger.info("Optimizing take-profit barriers...")
    best_tp, barrier_results = labeler.optimize_barriers(
        df.copy(),
        tp_candidates=config['labeling']['take_profit_atr_multiples']
    )
    
    # Apply optimal labeling
    labeler.tp_atr = best_tp
    df = labeler.label_data(df)
    
    # ========== FEATURE ENGINEERING ==========
    logger.info("\n" + "="*60)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("="*60)
    
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.engineer_features(df)
    
    logger.info(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} features")
    
    # ========== MODEL TRAINING ==========
    logger.info("\n" + "="*60)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("="*60)
    
    # Split data for initial training
    train_df, val_df, test_df = data_loader.split_train_val_test(
        df,
        config['validation']['train_months'],
        config['validation']['val_months'],
        config['validation']['test_months']
    )
    
    # Prepare data
    trainer = ModelTrainer(config)
    X_train, y_train, feature_names = trainer.prepare_data(train_df)
    X_val, y_val, _ = trainer.prepare_data(val_df)
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Features: {len(feature_names)}")
    
    # Train all models
    optimize_hp = config['hyperparameter_tuning'].get('method') == 'optuna'
    models = trainer.train_all_models(
        X_train, y_train, X_val, y_val,
        feature_names,
        optimize_hp=optimize_hp
    )
    
    logger.info(f"Trained {len(models)} models successfully")
    
    # ========== WALK-FORWARD VALIDATION ==========
    logger.info("\n" + "="*60)
    logger.info("STEP 5: WALK-FORWARD VALIDATION")
    logger.info("="*60)
    
    validator = WalkForwardValidator(config)
    results_df = validator.run_walk_forward(models, df, feature_names, trainer.scalers)
    
    # Check deployment readiness
    deployment_summary = validator.check_deployment_ready(
        results_df,
        config['performance_targets']
    )
    
    # ========== RISK MANAGEMENT BACKTESTING ==========
    logger.info("\n" + "="*60)
    logger.info("STEP 6: RISK-ADJUSTED BACKTESTING")
    logger.info("="*60)
    
    risk_manager = RiskManager(config)
    
    # Backtest best model with position sizing
    best_model_name = deployment_summary.index[0]
    best_model = models[best_model_name]
    
    logger.info(f"Running risk-adjusted backtest for: {best_model_name}")
    
    X_test, y_test, _ = trainer.prepare_data(test_df)
    y_pred_proba = validator.predict_model(
        best_model, best_model_name, X_test,
        trainer.scalers.get(best_model_name)
    )
    y_pred = (y_pred_proba >= config['risk_management']['execution_filters']['min_confidence']).astype(int)
    
    # Apply execution filters
    y_pred_filtered = risk_manager.apply_execution_filters(test_df, y_pred, y_pred_proba)
    
    # Run backtest with sizing
    equity_curve, backtest_summary = risk_manager.backtest_with_sizing(
        test_df[test_df['label'] != 0],
        y_pred_filtered[test_df['label'] != 0],
        account_value=10000.0,
        sl_atr_multiple=config['labeling']['stop_loss_atr_multiple']
    )
    
    logger.info("Backtest Results:")
    for metric, value in backtest_summary.items():
        logger.info(f"  {metric}: {value}")
    
    # ========== EXPORT RESULTS ==========
    logger.info("\n" + "="*60)
    logger.info("STEP 7: EXPORTING MODELS & REPORTS")
    logger.info("="*60)
    
    exporter = ModelExporter(config['paths']['output'])
    
    # Save all models
    for model_name, model in models.items():
        exporter.save_model(model, model_name, trainer.scalers.get(model_name))
        
        if config['deployment']['export_onnx']:
            exporter.export_to_onnx(model, model_name, len(feature_names))
    
    # Generate reports
    report = exporter.generate_performance_report(results_df, deployment_summary)
    
    # Generate plots
    exporter.plot_feature_importance(trainer.feature_importance)
    exporter.plot_equity_curves(results_df)
    exporter.plot_return_distribution(results_df)
    
    # Create deployment config
    deployment_config = exporter.create_deployment_config(
        best_model_name,
        feature_names,
        config
    )
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*60)
    
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"Deployment Ready: {deployment_summary.loc[best_model_name, 'deployment_ready']:.1%} of folds")
    logger.info(f"\nAverage Test Performance:")
    logger.info(f"  Profit Factor: {deployment_summary.loc[best_model_name, 'profit_factor']:.2f}")
    logger.info(f"  Win Rate: {deployment_summary.loc[best_model_name, 'win_rate']:.2%}")
    logger.info(f"  Max DD: {deployment_summary.loc[best_model_name, 'max_drawdown_pct']:.2f}%")
    logger.info(f"  Sharpe: {deployment_summary.loc[best_model_name, 'sharpe']:.2f}")
    logger.info(f"  R-multiple: {deployment_summary.loc[best_model_name, 'r_multiple']:.2f}")
    
    logger.info(f"\nOutputs saved to: {config['paths']['output']}")
    logger.info("  - Models: ./models/")
    logger.info("  - Reports: ./reports/")
    logger.info("  - Plots: ./plots/")
    logger.info("  - Config: ./configs/")
    
    # Check if models meet deployment criteria
    if deployment_summary.loc[best_model_name, 'deployment_ready'] >= 0.5:
        logger.info("\n✅ DEPLOYMENT CRITERIA MET - Model ready for production")
    else:
        logger.warning("\n⚠️  DEPLOYMENT CRITERIA NOT MET - Review and retrain")
        logger.warning("Common issues:")
        logger.warning("  - Insufficient training data")
        logger.warning("  - Suboptimal feature engineering")
        logger.warning("  - Data leakage (check lagging)")
        logger.warning("  - Overly aggressive performance targets")
    
    logger.info("\n" + "="*60)
    logger.info("Next Steps:")
    logger.info("1. Review performance report: ./reports/performance_report.md")
    logger.info("2. Examine feature importance plots")
    logger.info("3. Check equity curves for stability")
    logger.info("4. Test on paper trading account")
    logger.info("5. Set up retraining schedule (weekly recommended)")
    logger.info("6. Monitor PSI for data drift")
    logger.info("="*60)
    
    return models, results_df, deployment_summary


if __name__ == '__main__':
    try:
        models, results, summary = main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)