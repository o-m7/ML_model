"""Ensemble classifiers blending multiple models."""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional
from .base import BaseModel


class EnsembleClassifier(BaseModel):
    """
    Ensemble of LightGBM, XGBoost, and Logistic Regression.
    
    Combines predictions via weighted average or stacking.
    """
    
    def __init__(
        self,
        n_classes: int = 3,
        method: str = "weighted",
        weights: Optional[dict] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            n_classes: Number of classes (default 3: Up/Down/Flat)
            method: 'weighted' or 'stacking'
            weights: Dict with 'lgb', 'xgb', 'linear' weights
        """
        super().__init__()
        self.n_classes = n_classes
        self.method = method
        self.weights = weights or {'lgb': 0.40, 'xgb': 0.40, 'linear': 0.20}
        self.models = {}
        self.scaler = StandardScaler()
        self.meta_model = None
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ):
        """Train all ensemble members."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Compute class weights if not provided
        if sample_weight is None:
            class_counts = np.bincount(y.astype(int))
            class_weights = len(y) / (len(class_counts) * class_counts)
            sample_weight = class_weights[y.astype(int)]
        
        # 1. LightGBM (optimized for speed)
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=50,  # Reduced from 150 for faster training
            max_depth=4,      # Reduced from 5
            learning_rate=0.1, # Increased from 0.05 for faster convergence
            num_leaves=15,    # Reduced from 31
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_samples=20,
            random_state=42,
            verbosity=-1,
            force_row_wise=True
        )
        self.models['lgb'].fit(X, y, sample_weight=sample_weight)
        
        # 2. XGBoost (optimized for speed)
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=50,  # Reduced from 150
            max_depth=4,      # Reduced from 5
            learning_rate=0.1, # Increased from 0.05
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.5,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=5,
            random_state=42,
            verbosity=0
        )
        
        eval_set = [(X, y), (X_val, y_val)] if X_val is not None else None
        self.models['xgb'].fit(
            X, y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False
        )
        
        # 3. Logistic Regression (optimized for speed)
        self.models['linear'] = LogisticRegression(
            penalty='l2',  # Changed from elasticnet for faster convergence
            solver='lbfgs',  # Changed from saga - faster
            C=0.1,
            max_iter=500,  # Reduced from 1000
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        self.models['linear'].fit(X_scaled, y)
        
        # 4. Stacking meta-model (if method == 'stacking')
        if self.method == 'stacking' and X_val is not None:
            # Get OOF predictions
            oof_preds = self._get_base_predictions(X, X_val)
            
            self.meta_model = LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=500,
                random_state=42
            )
            self.meta_model.fit(oof_preds, y_val)
        
        self.is_fitted = True
    
    def _get_base_predictions(self, X: np.ndarray, X_val: np.ndarray) -> np.ndarray:
        """Get predictions from base models for stacking."""
        X_scaled = self.scaler.transform(X)
        
        preds = []
        for name in ['lgb', 'xgb', 'linear']:
            if name == 'linear':
                pred = self.models[name].predict_proba(X_scaled)
            else:
                pred = self.models[name].predict_proba(X)
            preds.append(pred)
        
        # Concatenate all predictions
        return np.hstack(preds)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'weighted':
            # Weighted average of base models
            preds = []
            for name, model in self.models.items():
                if name == 'linear':
                    pred = model.predict_proba(X_scaled)
                else:
                    pred = model.predict_proba(X)
                preds.append(pred * self.weights[name])
            
            ensemble_pred = np.sum(preds, axis=0)
            
        elif self.method == 'stacking':
            # Stacking with meta-model
            base_preds = self._get_base_predictions(X, X)
            ensemble_pred = self.meta_model.predict_proba(base_preds)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return ensemble_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_feature_importance(self, feature_names: list) -> dict:
        """Get feature importance from tree models."""
        importance_lgb = self.models['lgb'].feature_importances_
        importance_xgb = self.models['xgb'].feature_importances_
        
        # Average importance
        avg_importance = (importance_lgb + importance_xgb) / 2
        
        return dict(zip(feature_names, avg_importance))

