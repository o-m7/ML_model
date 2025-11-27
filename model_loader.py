"""
Model Loader - Load NEWEST trained ML models for live signal generation
Source: citadel_v6.py (OHLCV) + citadel_quote_models.py (Quote)
Supports: 1T, 5T, 15T, 30T only
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage trained ML models from citadel training systems"""
    
    def __init__(self, base_dir: str = "/Users/omar/Desktop/ML_model/ML_model"):
        self.base_dir = base_dir
        self.ohlcv_dir = os.path.join(base_dir, "OHLCV_models")
        self.quote_dir = os.path.join(base_dir, "QUOTE_models")
        self.models = {}
        self.timeframes = ["1T", "5T", "15T", "30T"]  # Only these timeframes
    
    def load_ohlcv_models(self) -> Dict[str, Any]:
        """Load all OHLCV models for all timeframes (1T, 5T, 15T, 30T)"""
        models = {}
        strategies = ["trend_following", "mean_reversion", "volatility_breakout"]
        
        for timeframe in self.timeframes:
            for strategy in strategies:
                # Model format: {strategy}_C_XAU-USD_{timeframe}.pkl
                model_name = f"{strategy}_C_XAU-USD_{timeframe}.pkl"
                model_path = os.path.join(self.ohlcv_dir, model_name)
                
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        models[f"{strategy}_{timeframe}"] = model
                        logger.info(f"✓ Loaded OHLCV {strategy} ({timeframe})")
                    except Exception as e:
                        logger.error(f"✗ Failed to load {model_name}: {e}")
                else:
                    logger.debug(f"⚠ Model not found: {model_path}")
        
        self.models.update(models)
        return models
    
    def load_quote_models(self) -> Dict[str, Any]:
        """Load Quote models for all timeframes (1T, 5T, 15T, 30T)"""
        models = {}
        strategies = ["mean_reversion", "volatility_breakout"]
        
        for timeframe in self.timeframes:
            for strategy in strategies:
                # Model format: {strategy}_C:XAU-USD_{timeframe}.pkl (note: colon not hyphen)
                model_name = f"{strategy}_C:XAU-USD_{timeframe}.pkl"
                model_path = os.path.join(self.quote_dir, model_name)
                
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        models[f"quote_{strategy}_{timeframe}"] = model
                        logger.info(f"✓ Loaded Quote {strategy} ({timeframe})")
                    except Exception as e:
                        logger.error(f"✗ Failed to load {model_name}: {e}")
                else:
                    logger.debug(f"⚠ Model not found: {model_path}")
        
        self.models.update(models)
        return models
    
    def load_all_models(self) -> Dict[str, Any]:
        """Load all newest trained models (from citadel_v6 and citadel_quote_models)"""
        logger.info("Loading all newest models (1T, 5T, 15T, 30T)...")
        
        # Load OHLCV models (citadel_v6)
        self.load_ohlcv_models()
        
        # Load Quote models (citadel_quote_models)
        self.load_quote_models()
        
        logger.info(f"✓ Loaded {len(self.models)} total models")
        return self.models
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a specific model by name"""
        return self.models.get(name)
    
    def predict(self, model_name: str, features: Dict[str, float]) -> Optional[float]:
        """Generate prediction from model"""
        model = self.get_model(model_name)
        if not model:
            logger.warning(f"Model {model_name} not found")
            return None
        
        try:
            # Models expect sklearn-compatible interface
            # Convert features dict to array matching training format
            if hasattr(model, 'predict'):
                # Get feature names from model if available
                feature_names = getattr(model, 'feature_names_in_', None)
                
                if feature_names is not None:
                    X = [[features.get(fname, 0.0) for fname in feature_names]]
                else:
                    # Fallback: assume features in sorted order
                    X = [[v for k, v in sorted(features.items())]]
                
                prediction = model.predict(X)[0]
                return float(prediction)
        except Exception as e:
            logger.error(f"Prediction error ({model_name}): {e}")
        
        return None


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    loader = ModelLoader()
    models = loader.load_all_models()
    
    print(f"\n{'─'*60}")
    print(f"LOADED MODELS:")
    print(f"{'─'*60}")
    for name, model in models.items():
        print(f"  ✓ {name}")
    print(f"{'─'*60}\n")
