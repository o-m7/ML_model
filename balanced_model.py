"""
BalancedModel class for loading XAGUSD models.

This module provides the BalancedModel class definition needed to unpickle
XAGUSD models that were trained with this wrapper.
"""

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler


class BalancedModel:
    """Balanced LightGBM model."""

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()

    def fit(self, X, y):
        """Train with balanced class weights."""

        X_scaled = self.scaler.fit_transform(X)

        # BALANCED weights (no excessive boosting)
        counts = np.bincount(y)
        weights = len(y) / (len(counts) * counts)
        weights[0] *= 1.5  # Moderate Flat boost
        if len(weights) > 2:
            weights[1] *= 1.2  # Slight Long boost
            weights[2] *= 1.2  # Slight Short boost (for balance)

        sample_weight = weights[y]

        self.model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.04,
            num_leaves=12,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=3.0,
            reg_lambda=4.0,
            min_child_samples=40,
            random_state=42,
            verbosity=-1,
            force_row_wise=True
        )

        self.model.fit(X_scaled, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
