"""Base model interface and utilities."""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    """Base interface for all models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class ModelCard:
    """Model metadata and tracking."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        strategy: str,
        features: List[str],
        training_window: Dict[str, str],
        label_config: Dict,
        cv_config: Dict,
        performance: Dict
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy
        self.features = features
        self.training_window = training_window
        self.label_config = label_config
        self.cv_config = cv_config
        self.performance = performance
        
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'strategy': self.strategy,
            'n_features': len(self.features),
            'features': self.features,
            'training_window': self.training_window,
            'label_config': self.label_config,
            'cv_config': self.cv_config,
            'performance': self.performance
        }
    
    def save(self, path: Path):
        """Save model card to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load model card from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

