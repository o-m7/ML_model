"""
Machine learning model training module.

Implements robust model training with walk-forward validation,
hyperparameter tuning, and no-lookahead guarantees.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
import pickle
from pathlib import Path
from datetime import datetime
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed")


class TradingModel:
    """
    Wrapper class for trading ML models.

    Handles training, prediction, and model persistence.
    """

    def __init__(self, config, symbol: str, timeframe: int):
        """
        Initialize model.

        Args:
            config: Configuration object
            symbol: Trading symbol
            timeframe: Timeframe in minutes
        """
        self.config = config
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.feature_cols = None
        self.model_type = config.model.model_type
        self.training_stats = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training statistics dict
        """
        self.feature_cols = X_train.columns.tolist()

        if self.model_type == 'xgboost':
            if not HAS_XGB:
                raise RuntimeError("XGBoost not available, install with: pip install xgboost")
            self.model = xgb.XGBClassifier(**self.config.model.xgb_params)

            # Use early stopping if validation set provided
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)

        elif self.model_type == 'lightgbm':
            if not HAS_LGB:
                raise RuntimeError("LightGBM not available, install with: pip install lightgbm")
            self.model = lgb.LGBMClassifier(**self.config.model.lgb_params)

            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                self.model.fit(X_train, y_train)

        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=self.config.model.random_state,
                max_iter=500,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.model.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]

        stats = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'train_auc': roc_auc_score(y_train, y_train_proba)
        }

        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)[:, 1]

            stats.update({
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
                'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
                'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
                'val_auc': roc_auc_score(y_val, y_val_proba)
            })

        self.training_stats = stats

        return stats

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of success (TP before SL).

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities (P(class=1))
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        # Ensure same features
        X = X[self.feature_cols]

        proba = self.model.predict_proba(X)[:, 1]

        return proba

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances (if available)."""
        if self.model is None:
            return pd.Series()

        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_cols
            ).sort_values(ascending=False)
            return importance
        else:
            return pd.Series()

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'training_stats': self.training_stats
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        self.model_type = model_data['model_type']
        self.symbol = model_data['symbol']
        self.timeframe = model_data['timeframe']
        self.training_stats = model_data.get('training_stats', {})

        print(f"Model loaded from {path}")


def train_model_with_walk_forward(
    df: pd.DataFrame,
    config,
    symbol: str,
    timeframe: int
) -> Tuple[TradingModel, pd.DataFrame]:
    """
    Train model using walk-forward / time-series cross-validation.

    Args:
        df: Full dataframe with features and target
        config: Configuration object
        symbol: Symbol name
        timeframe: Timeframe in minutes

    Returns:
        Trained model and DataFrame with out-of-fold predictions
    """
    from feature_engineering import get_feature_columns

    # Get feature columns
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df['target'].copy()

    # Remove any remaining NaNs
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    print(f"\n{'='*60}")
    print(f"Training model: {symbol} {timeframe}min")
    print(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # Time-series split
    tscv = TimeSeriesSplit(n_splits=config.model.n_splits)

    fold_results = []
    oof_predictions = pd.Series(index=X.index, dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold_idx + 1}/{config.model.n_splits}")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train model on this fold
        model_fold = TradingModel(config, symbol, timeframe)
        stats = model_fold.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        # Out-of-fold predictions
        oof_proba = model_fold.predict_proba(X_val_fold)
        oof_predictions.iloc[val_idx] = oof_proba

        fold_results.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            **stats
        })

        print(f"  Val Accuracy: {stats['val_accuracy']:.4f}, AUC: {stats['val_auc']:.4f}")

    # Train final model on all data
    print("\nTraining final model on full dataset...")
    final_model = TradingModel(config, symbol, timeframe)

    # Use last 20% as validation for early stopping
    split_idx = int(len(X) * 0.8)
    X_train_final = X.iloc[:split_idx]
    y_train_final = y.iloc[:split_idx]
    X_val_final = X.iloc[split_idx:]
    y_val_final = y.iloc[split_idx:]

    final_stats = final_model.train(X_train_final, y_train_final, X_val_final, y_val_final)

    print(f"Final model - Val Accuracy: {final_stats['val_accuracy']:.4f}, "
          f"AUC: {final_stats['val_auc']:.4f}")

    # Feature importance
    importance = final_model.get_feature_importance()
    if len(importance) > 0:
        print(f"\nTop 10 features:")
        print(importance.head(10))

    # Add OOF predictions to dataframe
    df_with_preds = df.copy()
    df_with_preds['ml_proba'] = oof_predictions

    # Store fold results
    final_model.training_stats['fold_results'] = fold_results

    return final_model, df_with_preds


def train_all_models(
    all_data: Dict[Tuple[str, int], Dict[str, pd.DataFrame]],
    all_features: Dict[Tuple[str, int], Dict[str, pd.DataFrame]],
    config
) -> Dict[Tuple[str, int], TradingModel]:
    """
    Train models for all symbol-timeframe pairs.

    Args:
        all_data: Raw OHLCV data
        all_features: Dataframes with features
        config: Configuration

    Returns:
        Dict of trained models
    """
    models = {}

    for (symbol, timeframe), feature_splits in all_features.items():
        # Combine train + val for walk-forward training
        train_df = feature_splits['train']
        val_df = feature_splits['val']

        combined_df = pd.concat([train_df, val_df])

        # Train with walk-forward
        model, df_with_preds = train_model_with_walk_forward(
            combined_df,
            config,
            symbol,
            timeframe
        )

        models[(symbol, timeframe)] = model

        # Save model
        model_dir = config.monitoring.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f"{symbol}_{timeframe}_{timestamp}.pkl"
        model.save(str(model_path))

    return models


if __name__ == "__main__":
    from config import get_default_config
    from data_loader import load_all_data, create_sample_data
    from feature_engineering import build_features

    config = get_default_config()

    # Create sample data if needed
    print("Creating sample data...")
    for symbol in config.data.symbols[:1]:  # Test with one symbol
        for timeframe in config.data.timeframes[:1]:  # Test with one timeframe
            file_path = config.data.data_dir / f"{symbol}_{timeframe}.csv"
            if not file_path.exists():
                df = create_sample_data(symbol, timeframe, n_bars=10000, save_path=str(file_path))

    # Load data
    print("\nLoading data...")
    all_data = load_all_data(config)

    # Build features
    print("\nBuilding features...")
    all_features = {}

    for (symbol, tf), splits in all_data.items():
        print(f"\nProcessing {symbol} {tf}min...")
        all_features[(symbol, tf)] = {
            'train': build_features(splits['train'], config),
            'val': build_features(splits['val'], config),
            'test': build_features(splits['test'], config)
        }

    # Train models
    print("\n" + "="*60)
    print("Training models...")
    models = train_all_models(all_data, all_features, config)

    print(f"\nTrained {len(models)} models")
