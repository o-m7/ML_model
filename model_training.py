"""
Machine learning model training module - FIXED VERSION.

Implements robust model training with:
- Class imbalance handling (scale_pos_weight)
- Proper metrics for trading (AUC, precision, recall)
- Probability calibration
- Walk-forward validation
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
    roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

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
    """Wrapper class for trading ML models with class imbalance handling."""

    def __init__(self, config, symbol: str, timeframe: int):
        self.config = config
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.calibrated_model = None
        self.feature_cols = None
        self.model_type = config.model.model_type
        self.training_stats = {}
        self.optimal_threshold = 0.5

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Train the model with class imbalance handling."""

        self.feature_cols = X_train.columns.tolist()

        # Calculate class weights
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        print(f"  Class distribution: {n_neg} negative, {n_pos} positive")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")

        if self.model_type == 'xgboost':
            if not HAS_XGB:
                raise RuntimeError("XGBoost not available")

            # Updated params for imbalanced data
            params = self.config.model.xgb_params.copy()
            params.update({
                'scale_pos_weight': scale_pos_weight,
                'max_depth': 4,
                'min_child_weight': 5,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 0.2,
                'reg_alpha': 0.3,
                'reg_lambda': 2.0,
                'eval_metric': 'auc',
            })

            self.model = xgb.XGBClassifier(**params)

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
                raise RuntimeError("LightGBM not available")

            params = self.config.model.lgb_params.copy()
            params.update({
                'scale_pos_weight': scale_pos_weight,
                'max_depth': 4,
                'min_child_samples': 30,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.3,
                'reg_lambda': 2.0,
                'metric': 'auc',
            })

            self.model = lgb.LGBMClassifier(**params)

            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                self.model.fit(X_train, y_train)

        else:
            # Logistic regression with class weight
            class_weight = {0: 1.0, 1: scale_pos_weight}
            self.model = LogisticRegression(
                class_weight=class_weight,
                random_state=self.config.model.random_state,
                max_iter=500,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

        # Calibrate probabilities
        print("  Calibrating probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method='isotonic',
            cv='prefit'
        )

        if X_val is not None and y_val is not None:
            self.calibrated_model.fit(X_val, y_val)
        else:
            split_idx = int(len(X_train) * 0.8)
            self.calibrated_model.fit(
                X_train.iloc[split_idx:],
                y_train.iloc[split_idx:]
            )

        # Compute metrics
        y_train_pred = self.predict(X_train)
        y_train_proba = self.predict_proba(X_train)

        stats = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'train_auc': roc_auc_score(y_train, y_train_proba) if len(np.unique(y_train)) > 1 else 0,
            'train_brier': brier_score_loss(y_train, y_train_proba)
        }

        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            y_val_proba = self.predict_proba(X_val)

            stats.update({
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
                'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
                'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
                'val_auc': roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0,
                'val_brier': brier_score_loss(y_val, y_val_proba)
            })

        self.training_stats = stats
        return stats

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes."""
        X = X[self.feature_cols]
        if self.calibrated_model is not None:
            return self.calibrated_model.predict(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probability of success."""
        X = X[self.feature_cols]

        if self.calibrated_model is not None:
            proba = self.calibrated_model.predict_proba(X)[:, 1]
        else:
            proba = self.model.predict_proba(X)[:, 1]

        return proba

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances."""
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

    def select_top_features(self, n_features: int = 40):
        """Select top N features by importance."""
        importance = self.get_feature_importance()

        if len(importance) == 0:
            return

        top_features = importance.head(n_features).index.tolist()
        self.feature_cols = top_features
        print(f"  Selected top {n_features} features")

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'training_stats': self.training_stats,
            'optimal_threshold': self.optimal_threshold
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.calibrated_model = model_data.get('calibrated_model')
        self.feature_cols = model_data['feature_cols']
        self.model_type = model_data['model_type']
        self.symbol = model_data['symbol']
        self.timeframe = model_data['timeframe']
        self.training_stats = model_data.get('training_stats', {})
        self.optimal_threshold = model_data.get('optimal_threshold', 0.5)


def tune_probability_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    df: pd.DataFrame,
    model,
    config,
    symbol: str,
    timeframe: int,
    thresholds: List[float] = None
) -> Tuple[float, Dict]:
    """
    Tune probability threshold to maximize trading metrics.
    """
    from backtest import Backtest
    from metrics import calculate_metrics

    if thresholds is None:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    print(f"\n  Tuning probability threshold...")
    print(f"  Probability distribution: min={y_proba.min():.3f}, "
          f"median={np.median(y_proba):.3f}, max={y_proba.max():.3f}")

    best_threshold = 0.5
    best_pf = 0
    best_metrics = {}

    for threshold in thresholds:
        original_threshold = config.strategy.ml_prob_threshold
        config.strategy.ml_prob_threshold = threshold

        df_test = df.copy()
        df_test['ml_proba'] = y_proba

        try:
            bt = Backtest(df_test, model, config, symbol, timeframe)
            results = bt.run()
            trade_log = bt.get_trade_log()

            if len(trade_log) == 0:
                print(f"    Threshold {threshold:.2f}: 0 trades")
                continue

            equity_curve = results['equity']
            metrics = calculate_metrics(equity_curve, trade_log, config.risk.initial_capital)

            if (metrics.total_trades >= 20 and
                abs(metrics.max_drawdown_pct) <= 8.0 and
                metrics.profit_factor > best_pf):

                best_pf = metrics.profit_factor
                best_threshold = threshold
                best_metrics = {
                    'trades': metrics.total_trades,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'max_dd_pct': metrics.max_drawdown_pct,
                    'sharpe': metrics.sharpe_ratio
                }

            print(f"    Threshold {threshold:.2f}: {metrics.total_trades} trades, "
                  f"PF={metrics.profit_factor:.2f}, WR={metrics.win_rate*100:.1f}%, "
                  f"DD={metrics.max_drawdown_pct:.1f}%")

        except Exception as e:
            print(f"    Threshold {threshold:.2f}: Error - {e}")

        config.strategy.ml_prob_threshold = original_threshold

    if best_metrics:
        print(f"  → Optimal threshold: {best_threshold:.2f} (PF={best_pf:.2f})")
    else:
        print(f"  → No valid threshold found, using default 0.50")
        best_threshold = 0.50

    return best_threshold, best_metrics


def train_model_with_walk_forward(
    df: pd.DataFrame,
    config,
    symbol: str,
    timeframe: int
) -> Tuple[TradingModel, pd.DataFrame]:
    """Train model using walk-forward / time-series cross-validation."""
    from feature_engineering import get_feature_columns

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df['target'].copy()

    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    print(f"\n{'='*60}")
    print(f"Training model: {symbol} {timeframe}min")
    print(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    tscv = TimeSeriesSplit(n_splits=config.model.n_splits)

    fold_results = []
    oof_predictions = pd.Series(index=X.index, dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold_idx + 1}/{config.model.n_splits}")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model_fold = TradingModel(config, symbol, timeframe)
        stats = model_fold.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        oof_proba = model_fold.predict_proba(X_val_fold)
        oof_predictions.iloc[val_idx] = oof_proba

        fold_results.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            **stats
        })

        print(f"  Val Accuracy: {stats['val_accuracy']:.4f}, AUC: {stats['val_auc']:.4f}")

    print("\nTraining final model on full dataset...")
    final_model = TradingModel(config, symbol, timeframe)

    split_idx = int(len(X) * 0.8)
    X_train_final = X.iloc[:split_idx]
    y_train_final = y.iloc[:split_idx]
    X_val_final = X.iloc[split_idx:]
    y_val_final = y.iloc[split_idx:]

    final_stats = final_model.train(X_train_final, y_train_final, X_val_final, y_val_final)

    print(f"Final model - Val Accuracy: {final_stats['val_accuracy']:.4f}, "
          f"AUC: {final_stats['val_auc']:.4f}")

    importance = final_model.get_feature_importance()
    if len(importance) > 0:
        print(f"\nTop 10 features:")
        print(importance.head(10))

    final_model.select_top_features(n_features=40)

    df_with_preds = df.copy()
    df_with_preds['ml_proba'] = oof_predictions

    final_model.training_stats['fold_results'] = fold_results

    return final_model, df_with_preds


def train_all_models(
    all_data: Dict[Tuple[str, int], Dict[str, pd.DataFrame]],
    all_features: Dict[Tuple[str, int], Dict[str, pd.DataFrame]],
    config
) -> Dict[Tuple[str, int], TradingModel]:
    """Train models for all symbol-timeframe pairs."""
    models = {}

    for (symbol, timeframe), feature_splits in all_features.items():
        train_df = feature_splits['train']
        val_df = feature_splits['val']

        combined_df = pd.concat([train_df, val_df])

        model, df_with_preds = train_model_with_walk_forward(
            combined_df,
            config,
            symbol,
            timeframe
        )

        models[(symbol, timeframe)] = model

        model_dir = config.monitoring.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f"{symbol}_{timeframe}_{timestamp}.pkl"
        model.save(str(model_path))

    return models
