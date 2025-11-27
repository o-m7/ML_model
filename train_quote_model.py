"""
Quote-Based ML Model Training Script
Trains a classification model using ONLY quote-derived microstructure features
for intraday trading signal generation.

No shuffling. Time-series aware. Production-ready.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data path - supports .parquet or .csv
DATA_PATH: str = "feature_store/C:XAU-USD/quotes/C:XAU-USD_5T_quotes_labeled.parquet"

# Target column name (binary classification label)
TARGET_COL: str = "label"

# All available quote features (will use those present in data)
FEATURE_COLS: List[str] = [
    # Core OHLCV
    "open", "high", "low", "close", "volume",
    
    # Quote price columns
    "bid_first", "bid_last", "bid_min", "bid_max",
    "ask_first", "ask_last", "ask_min", "ask_max",
    
    # Aggregated mid/depth/stats
    "mid_price_mean", "mid_price_std",
    "bid_size_sum", "bid_size_mean", "bid_size_max",
    "ask_size_sum", "ask_size_mean", "ask_size_max",
    
    # Spread & pressure
    "spread", "spread_pct", "buy_pressure", "sell_pressure",
    
    # Added microstructure features (12 critical)
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

# Model parameters
TRAIN_RATIO: float = 0.7
RANDOM_STATE: int = 42

# Artifact directory
ARTIFACTS_DIR: str = "artifacts"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from parquet or CSV file.
    
    Args:
        path: Path to data file (.parquet or .csv)
    
    Returns:
        Cleaned dataframe with timestamp parsed and sorted
    
    Raises:
        ValueError: If file format not supported or file not found
    """
    logger.info(f"Loading data from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # Detect file format
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format. Use .parquet or .csv")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Parse timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Sorted by timestamp: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Drop rows with missing label
    if TARGET_COL in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=[TARGET_COL])
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing label")
    
    logger.info(f"Final dataset: {len(df)} rows")
    return df


def select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Select only the intersection of available columns and requested features.
    
    Args:
        df: Input dataframe
        feature_cols: List of requested feature column names
        target_col: Name of target column
    
    Returns:
        X (features dataframe), y (target series), available_features (list of used columns)
    
    Raises:
        ValueError: If no features found or target column missing
    """
    logger.info("\n" + "="*70)
    logger.info("FEATURE SELECTION")
    logger.info("="*70)
    
    # Check target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Find available features
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if len(available_features) == 0:
        raise ValueError("No requested features found in dataframe")
    
    logger.info(f"✓ Found {len(available_features)} features")
    if missing_features:
        logger.warning(f"⚠ Missing {len(missing_features)} features:")
        for feat in missing_features[:10]:
            logger.warning(f"    - {feat}")
        if len(missing_features) > 10:
            logger.warning(f"    ... and {len(missing_features) - 10} more")
    
    # Extract X and y
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Drop rows with NaN in any feature
    initial_rows = len(X)
    X = X.dropna()
    y = y.loc[X.index]
    dropped = initial_rows - len(X)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing feature values")
    
    logger.info(f"✓ Final feature matrix: {X.shape}")
    logger.info(f"✓ Label distribution: {y.value_counts().to_dict()}")
    
    return X, y, available_features


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform time-based train/validation split WITHOUT shuffling.
    
    Args:
        X: Feature dataframe (already sorted by time)
        y: Target series
        train_ratio: Fraction of data to use for training (0-1)
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    logger.info("\n" + "="*70)
    logger.info("TIME-BASED SPLIT (NO SHUFFLING)")
    logger.info("="*70)
    
    split_idx = int(len(X) * train_ratio)
    
    X_train = X.iloc[:split_idx].copy()
    X_val = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_val = y.iloc[split_idx:].copy()
    
    logger.info(f"Train set: {len(X_train)} samples ({100*train_ratio:.1f}%)")
    logger.info(f"  Labels: {y_train.value_counts().to_dict()}")
    logger.info(f"Val set: {len(X_val)} samples ({100*(1-train_ratio):.1f}%)")
    logger.info(f"  Labels: {y_val.value_counts().to_dict()}")
    
    return X_train, X_val, y_train, y_val


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data and transform both sets.
    
    Args:
        X_train: Training features
        X_val: Validation features
    
    Returns:
        X_train_scaled, X_val_scaled, fitted_scaler
    """
    logger.info("\n" + "="*70)
    logger.info("FEATURE SCALING")
    logger.info("="*70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logger.info("✓ StandardScaler fitted on training data")
    logger.info(f"✓ Transformed {len(X_train_scaled)} train + {len(X_val_scaled)} val samples")
    
    return X_train_scaled, X_val_scaled, scaler


def build_model() -> xgb.XGBClassifier:
    """
    Build XGBoost classifier with sensible defaults for trading data.
    
    Returns:
        Configured XGBClassifier
    """
    logger.info("\n" + "="*70)
    logger.info("BUILDING MODEL")
    logger.info("="*70)
    
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
    
    logger.info("✓ XGBClassifier configured")
    logger.info("  n_estimators=300")
    logger.info("  max_depth=6")
    logger.info("  learning_rate=0.05")
    logger.info("  subsample=0.8, colsample_bytree=0.8")
    
    return model


def train_model(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: pd.Series
) -> xgb.XGBClassifier:
    """
    Train the model on training data.
    
    Args:
        model: Unfitted XGBClassifier
        X_train: Training features (scaled)
        y_train: Training labels
    
    Returns:
        Fitted model
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING MODEL")
    logger.info("="*70)
    
    model.fit(X_train, y_train, verbose=False)
    
    logger.info("✓ Model training complete")
    
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series
) -> None:
    """
    Evaluate model performance on train and validation sets.
    
    Args:
        model: Fitted XGBClassifier
        X_train: Training features (scaled)
        y_train: Training labels
        X_val: Validation features (scaled)
        y_val: Validation labels
    """
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Training metrics
    logger.info("\n--- TRAINING SET ---")
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    logger.info(f"Accuracy:  {train_acc:.4f}")
    logger.info(f"Precision: {train_prec:.4f}")
    logger.info(f"Recall:    {train_rec:.4f}")
    logger.info(f"F1-Score:  {train_f1:.4f}")
    logger.info(f"ROC-AUC:   {train_auc:.4f}")
    
    # Validation metrics
    logger.info("\n--- VALIDATION SET ---")
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    logger.info(f"Accuracy:  {val_acc:.4f}")
    logger.info(f"Precision: {val_prec:.4f}")
    logger.info(f"Recall:    {val_rec:.4f}")
    logger.info(f"F1-Score:  {val_f1:.4f}")
    logger.info(f"ROC-AUC:   {val_auc:.4f}")
    
    # Confusion matrices
    logger.info("\n--- CONFUSION MATRICES ---")
    logger.info("Training Set:")
    logger.info(confusion_matrix(y_train, y_train_pred))
    logger.info("\nValidation Set:")
    logger.info(confusion_matrix(y_val, y_val_pred))


def display_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    top_n: int = 20
) -> None:
    """
    Display top N most important features.
    
    Args:
        model: Fitted XGBClassifier
        feature_names: List of feature column names (in order)
        top_n: Number of top features to display
    """
    logger.info("\n" + "="*70)
    logger.info(f"TOP {top_n} FEATURE IMPORTANCES")
    logger.info("="*70)
    
    importances = model.feature_importances_
    
    # Create feature importance pairs
    feature_imp = list(zip(feature_names, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    
    # Display top features
    for i, (feat, imp) in enumerate(feature_imp[:top_n], 1):
        bar = "█" * int(imp * 50)
        logger.info(f"{i:2d}. {feat:30s} {imp:7.4f}  {bar}")


def save_artifacts(
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    feature_names: List[str],
    directory: str = ARTIFACTS_DIR
) -> None:
    """
    Save trained model, scaler, and feature names to disk.
    
    Args:
        model: Fitted XGBClassifier
        scaler: Fitted StandardScaler
        feature_names: List of feature column names used
        directory: Output directory for artifacts
    """
    logger.info("\n" + "="*70)
    logger.info("SAVING ARTIFACTS")
    logger.info("="*70)
    
    # Create directory
    Path(directory).mkdir(exist_ok=True)
    
    # Save model
    model_path = os.path.join(directory, 'quote_model_xgb.pkl')
    joblib.dump(model, model_path)
    logger.info(f"✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(directory, 'quote_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(directory, 'quote_features.json')
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"✓ Feature names saved: {features_path}")
    
    logger.info(f"\n✓ All artifacts saved to: {directory}")


def main():
    """Main training pipeline."""
    logger.info("\n" + "#"*70)
    logger.info("# QUOTE-BASED ML MODEL TRAINING PIPELINE")
    logger.info("#"*70)
    
    try:
        # Load data
        df = load_data(DATA_PATH)
        
        # Select features
        X, y, available_features = select_features(df, FEATURE_COLS, TARGET_COL)
        
        # Time-based split
        X_train, X_val, y_train, y_val = time_series_split(X, y, train_ratio=TRAIN_RATIO)
        
        # Scale features
        X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
        
        # Build model
        model = build_model()
        
        # Train model
        model = train_model(model, X_train_scaled, y_train)
        
        # Evaluate model
        evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Feature importance
        display_feature_importance(model, available_features, top_n=20)
        
        # Save artifacts
        save_artifacts(model, scaler, available_features, directory=ARTIFACTS_DIR)
        
        logger.info("\n" + "#"*70)
        logger.info("✅ TRAINING PIPELINE COMPLETE")
        logger.info("#"*70)
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
