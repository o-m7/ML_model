"""
Batch Train Quote Models for All Timeframes
Trains XGBoost models on 1T, 5T, 15T, and 30T quote data
"""

import os
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Feature columns (same across all timeframes)
FEATURE_COLS: List[str] = [
    "open", "high", "low", "close", "volume",
    "bid_first", "bid_last", "bid_min", "bid_max",
    "ask_first", "ask_last", "ask_min", "ask_max",
    "mid_price_mean", "mid_price_std",
    "bid_size_sum", "bid_size_mean", "bid_size_max",
    "ask_size_sum", "ask_size_mean", "ask_size_max",
    "spread", "spread_pct", "buy_pressure", "sell_pressure",
    "microprice",
    "microprice_return_1", "microprice_return_5", "microprice_return_15",
    "quote_imbalance",
    "bid_return", "ask_return", "mid_return",
    "bid_vol", "ask_vol", "mid_vol",
    "spread_zscore", "spread_volatility", "spread_skew",
    "orderflow_volatility",
    "pressure_gradient",
    "volume_imbalance",
    "bid_liquidity_shock", "ask_liquidity_shock",
    "microprice_momentum", "midprice_momentum",
    "bid_queue_shift", "ask_queue_shift",
    "bid_size_shift", "ask_size_shift",
    "quote_atr_5", "quote_atr_10", "quote_atr_20",
    "quote_hl_range",
]

TARGET_COL: str = "label"
TRAIN_RATIO: float = 0.7
RANDOM_STATE: int = 42
ARTIFACTS_DIR: str = "artifacts"


def train_quote_model(timeframe: str) -> Dict:
    """Train quote model for a specific timeframe"""
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING QUOTE MODEL: {timeframe}")
    logger.info(f"{'='*70}")
    
    # Load data
    data_path = f"feature_store/C:XAU-USD/quotes/C:XAU-USD_{timeframe}_quotes_labeled.parquet"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Select features
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    logger.info(f"Using {len(available_features)} features")
    
    if TARGET_COL not in df.columns:
        logger.error(f"Target column '{TARGET_COL}' not found")
        return None
    
    X = df[available_features].copy()
    y = df[TARGET_COL].copy()
    
    # Drop NaN
    X = X.dropna()
    y = y.loc[X.index]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Time-based split
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]
    
    logger.info(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Build and train model
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    
    logger.info("Training XGBoost model...")
    model.fit(X_train_scaled, y_train, verbose=False)
    logger.info("✓ Training complete")
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_train, y_train_proba),
    }
    
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, zero_division=0),
        'recall': recall_score(y_val, y_val_pred, zero_division=0),
        'f1': f1_score(y_val, y_val_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_val_proba),
    }
    
    logger.info("\nTRAIN METRICS:")
    for metric, value in train_metrics.items():
        logger.info(f"  {metric:12s}: {value:.4f}")
    
    logger.info("\nVALIDATION METRICS:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric:12s}: {value:.4f}")
    
    # Save artifacts
    Path(ARTIFACTS_DIR).mkdir(exist_ok=True)
    
    model_path = os.path.join(ARTIFACTS_DIR, f'quote_model_{timeframe}_xgb.pkl')
    scaler_path = os.path.join(ARTIFACTS_DIR, f'quote_scaler_{timeframe}.pkl')
    features_path = os.path.join(ARTIFACTS_DIR, f'quote_features_{timeframe}.txt')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(available_features))
    
    logger.info(f"\n✓ Model saved: {model_path}")
    logger.info(f"✓ Scaler saved: {scaler_path}")
    logger.info(f"✓ Features saved: {features_path}")
    
    return {
        'timeframe': timeframe,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'n_features': len(available_features),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
    }


def main():
    """Train all timeframes"""
    logger.info("\n" + "#"*70)
    logger.info("# BATCH TRAIN QUOTE MODELS - ALL TIMEFRAMES")
    logger.info("#"*70)
    
    timeframes = ['1T', '5T', '15T', '30T']
    results = []
    
    for tf in timeframes:
        try:
            result = train_quote_model(tf)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {tf}: {e}", exc_info=True)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    
    logger.info("\n{:<6} {:<10} {:<10} {:<10} {:<10}".format(
        "TF", "VAL ACC", "VAL PREC", "VAL RECALL", "VAL AUC"
    ))
    logger.info("-"*70)
    
    for result in results:
        tf = result['timeframe']
        acc = result['val_metrics']['accuracy']
        prec = result['val_metrics']['precision']
        recall = result['val_metrics']['recall']
        auc = result['val_metrics']['roc_auc']
        
        logger.info("{:<6} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            tf, acc, prec, recall, auc
        ))
    
    logger.info("\n" + "#"*70)
    logger.info("✅ ALL MODELS TRAINED")
    logger.info("#"*70)


if __name__ == '__main__':
    main()
