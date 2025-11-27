"""
Model Converter: XGBoost Model Verification
============================================

Verifies that trained XGBoost models load correctly.
Uses native XGBoost inference instead of ONNX conversion.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import xgboost as xgb

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class ModelConverter:
    """Converts ML models to ONNX format."""
    
    def __init__(
        self,
        models_dir: str = "artifacts",
        onnx_output_dir: str = "artifacts/onnx_models",
    ):
        """
        Args:
            models_dir: Directory with joblib models
            onnx_output_dir: Output directory for ONNX models
        """
        self.models_dir = Path(models_dir)
        self.onnx_output_dir = Path(onnx_output_dir)
        self.onnx_output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_xgboost_to_onnx(
        self,
        model_path: str,
        output_path: str,
        initial_types: Optional[list] = None,
    ) -> bool:
        """
        Verify XGBoost model loads correctly (skipping ONNX conversion).
        
        Uses native XGBoost inference instead of ONNX to avoid compatibility issues.
        
        Args:
            model_path: Path to joblib-saved XGBoost model
            output_path: Not used (for compatibility)
            initial_types: Not used (for compatibility)
            
        Returns:
            True if model loads successfully
        """
        try:
            logger.info(f"Verifying XGBoost model: {model_path}")
            
            # Load model
            model = joblib.load(model_path)
            
            if not isinstance(model, xgb.XGBClassifier):
                logger.error("Model is not XGBClassifier")
                return False
            
            # Get feature count
            n_features = model.n_features_in_
            logger.debug(f"  ✓ Model loaded successfully ({n_features} features)")
            
            # Test inference with dummy data
            dummy_data = np.random.rand(1, n_features).astype(np.float32)
            try:
                probs = model.predict_proba(dummy_data)
                logger.debug(f"  ✓ Inference test passed (output shape: {probs.shape})")
            except Exception as e:
                logger.error(f"  ✗ Inference test failed: {e}")
                return False
            
            logger.info(f"✓ Model verified: {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def convert_all_models(self) -> int:
        """
        Verify all models in models_dir load correctly.
        
        Returns:
            Number of successfully verified models
        """
        verified_count = 0
        
        # Find all joblib models
        for model_file in self.models_dir.glob("*_xgb.pkl"):
            try:
                # Verify model
                if self.convert_xgboost_to_onnx(str(model_file), ""):
                    verified_count += 1
            
            except Exception as e:
                logger.error(f"Error processing {model_file}: {e}")
        
        logger.info(f"✓ Verified {verified_count} models")
        return verified_count
    
    def verify_onnx_model(self, onnx_path: str) -> bool:
        """
        Not used (ONNX conversion skipped).
        """
        logger.info(f"ONNX verification skipped (using native XGBoost)")
        return True
    
    def test_onnx_inference(
        self,
        onnx_path: str,
        test_features: Optional[np.ndarray] = None,
        n_features: int = 17,
    ) -> bool:
        """
        Not used (ONNX conversion skipped).
        """
        logger.info(f"ONNX inference test skipped (using native XGBoost)")
        return True


def batch_convert_models():
    """Verify all available XGBoost models."""
    logging.basicConfig(level=logging.INFO)
    
    converter = ModelConverter()
    
    logger.info("=" * 80)
    logger.info("MODEL VERIFICATION: XGBoost Native")
    logger.info("=" * 80)
    
    # Verify all models
    count = converter.convert_all_models()
    
    if count > 0:
        logger.info(f"\n✓ Successfully verified {count} models")
        logger.info("Using native XGBoost inference (no ONNX conversion needed)")
    else:
        logger.warning("No models found to verify")
    
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    batch_convert_models()
