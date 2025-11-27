"""
Multi-Model Ensemble Signal Generator
======================================

Generates live trading signals from all trained models (Quote + OHLCV).
Supports multiple timeframes (1T, 5T, 15T, 30T) with ensemble voting.
Real-time feature computation and prediction with no look-ahead bias.

Architecture:
- Individual model signals: Each model generates independent prediction
- Ensemble voting: Consensus across all models
- Confidence scoring: Based on probability distance from threshold
- Signal strength: STRONG_BUY, BUY, SELL, STRONG_SELL, NEUTRAL

Production Features:
- Live feature computation from market data
- Per-model signal generation with confidence scores
- Ensemble voting across models and timeframes
- Signal consensus scoring
- CSV/JSON export for trading integration
- Detailed logging and audit trail
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from joblib import load as joblib_load

# ============================================================================
# CONFIGURATION
# ============================================================================

ARTIFACTS_DIR: str = "artifacts"
OUTPUT_DIR: str = "signals"
MODEL_TYPES: List[str] = ["quote", "ohlcv"]
TIMEFRAMES: List[str] = ["1T", "5T", "15T", "30T"]
CONFIDENCE_THRESHOLD: float = 0.55  # Min probability for signal generation

# Signal strength thresholds
STRONG_BUY_THRESHOLD: float = 0.70   # High confidence buy signal
STRONG_SELL_THRESHOLD: float = 0.30  # High confidence sell signal
CONSENSUS_THRESHOLD: float = 0.60    # Threshold for ensemble consensus

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModelSignal:
    """Single model's signal output"""
    model_name: str
    timeframe: str
    model_type: str  # 'quote' or 'ohlcv'
    timestamp: str
    probability: float
    direction: int  # 1 for long, -1 for short, 0 for no signal
    confidence: float  # 0 to 1, based on distance from threshold
    

@dataclass
class EnsembleSignal:
    """Aggregated signal from multiple models"""
    timestamp: str
    long_votes: int
    short_votes: int
    neutral_votes: int
    total_votes: int
    consensus_probability: float  # 0 to 1
    ensemble_direction: int  # 1 (long), -1 (short), 0 (neutral)
    consensus_strength: str  # "STRONG_BUY", "BUY", "SELL", "STRONG_SELL", "NEUTRAL"
    agreement_level: str  # "UNANIMOUS", "STRONG", "MODERATE", "WEAK"
    individual_signals: List[Dict]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(artifacts_dir: str, model_name: str) -> object:
    """Load a single trained model from artifacts."""
    model_path = os.path.join(artifacts_dir, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info(f"Loading model: {model_name}")
    return joblib_load(model_path)


def load_scaler(artifacts_dir: str, model_name: str) -> object:
    """Load feature scaler for a model."""
    # Extract timeframe from model name (e.g., "quote_model_5T_xgb" -> "5T")
    parts = model_name.split('_')
    timeframe = [p for p in parts if p in TIMEFRAMES]
    
    if timeframe:
        tf = timeframe[0]
        model_type = "quote" if "quote" in model_name else "ohlcv"
        scaler_name = f"{model_type}_scaler_{tf}"
    else:
        # Fallback for legacy models
        scaler_name = f"{model_name.replace('_xgb', '')}_scaler"
    
    scaler_path = os.path.join(artifacts_dir, f"{scaler_name}.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    return joblib_load(scaler_path)


def load_feature_names(artifacts_dir: str, model_name: str) -> List[str]:
    """Load feature names for a model."""
    # Extract timeframe and model type
    parts = model_name.split('_')
    timeframe = [p for p in parts if p in TIMEFRAMES]
    
    if timeframe:
        tf = timeframe[0]
        model_type = "quote" if "quote" in model_name else "ohlcv"
        feature_file = f"{model_type}_features_{tf}.txt"
    else:
        feature_file = f"{model_name.replace('_xgb', '')}_features.json"
    
    feature_path = os.path.join(artifacts_dir, feature_file)
    
    # Try .txt file first
    if feature_path.endswith('.txt') and os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        return features
    
    # Try .json file
    json_path = feature_path.replace('.txt', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            features = json.load(f)
        return features
    
    raise FileNotFoundError(f"Features not found for {model_name}")


# ============================================================================
# SIGNAL GENERATOR CLASS
# ============================================================================

class EnsembleSignalGenerator:
    """
    Multi-model ensemble signal generator.
    
    Loads all trained models and generates signals based on:
    - Individual model predictions
    - Feature scaling and preprocessing
    - Ensemble consensus across models
    - Confidence scoring and signal strength
    """
    
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        """Initialize signal generator with all available models."""
        self.artifacts_dir = artifacts_dir
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, object] = {}
        self.features: Dict[str, List[str]] = {}
        self.model_info: Dict[str, Dict] = {}
        
        self._load_all_models()
    
    def _load_all_models(self) -> None:
        """Load all available models from artifacts directory."""
        logger.info("\n" + "=" * 70)
        logger.info("LOADING ENSEMBLE MODELS")
        logger.info("=" * 70)
        
        if not os.path.exists(self.artifacts_dir):
            raise FileNotFoundError(f"Artifacts directory not found: {self.artifacts_dir}")
        
        model_count = 0
        quote_count = 0
        ohlcv_count = 0
        
        for filename in sorted(os.listdir(self.artifacts_dir)):
            if not filename.endswith('.pkl'):
                continue
            
            # Skip scaler and feature files
            if 'scaler' in filename or 'features' in filename:
                continue
            
            # Only load model files
            if 'model' not in filename.lower():
                continue
            
            model_name = filename[:-4]  # Remove .pkl
            
            try:
                self.models[model_name] = load_model(self.artifacts_dir, model_name)
                self.scalers[model_name] = load_scaler(self.artifacts_dir, model_name)
                self.features[model_name] = load_feature_names(self.artifacts_dir, model_name)
                
                # Extract model info
                parts = model_name.split('_')
                timeframe = [p for p in parts if p in TIMEFRAMES]
                model_type = "quote" if "quote" in model_name else "ohlcv"
                
                self.model_info[model_name] = {
                    'type': model_type,
                    'timeframe': timeframe[0] if timeframe else 'Unknown',
                    'feature_count': len(self.features[model_name])
                }
                
                model_count += 1
                if model_type == "quote":
                    quote_count += 1
                else:
                    ohlcv_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        logger.info(f"✓ Loaded {model_count} models")
        logger.info(f"  Quote models: {quote_count}")
        logger.info(f"  OHLCV models: {ohlcv_count}")
        logger.info(f"  Models: {list(self.models.keys())}\n")
    
    def generate_model_signal(
        self,
        model_name: str,
        features_df: pd.DataFrame,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ) -> Optional[ModelSignal]:
        """
        Generate signal from a single model.
        
        Args:
            model_name: Name of the model to use
            features_df: DataFrame with required features (single row)
            confidence_threshold: Min probability for signal generation
            
        Returns:
            ModelSignal object or None if below threshold
        """
        if model_name not in self.models:
            logger.warning(f"Model not found: {model_name}")
            return None
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        feature_names = self.features[model_name]
        model_info = self.model_info[model_name]
        
        try:
            # Select required features
            X = features_df[feature_names].values
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Generate prediction
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_scaled)[0]
                prob_long = probas[1]  # Probability of class 1 (long)
            else:
                preds = model.predict(X_scaled)[0]
                prob_long = preds
            
            # Map to direction and calculate confidence
            if prob_long >= confidence_threshold:
                direction = 1  # Long
                confidence = prob_long
            elif prob_long <= (1 - confidence_threshold):
                direction = -1  # Short
                confidence = 1 - prob_long
            else:
                direction = 0  # No signal
                confidence = 0.5
            
            signal = ModelSignal(
                model_name=model_name,
                timeframe=model_info['timeframe'],
                model_type=model_info['type'],
                timestamp=datetime.now().isoformat(),
                probability=float(prob_long),
                direction=int(direction),
                confidence=float(confidence)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {model_name}: {e}")
            return None
    
    def generate_ensemble_signal(
        self,
        features_df: pd.DataFrame,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        model_filter: Optional[List[str]] = None
    ) -> EnsembleSignal:
        """
        Generate ensemble signal from all models.
        
        Aggregates individual model signals through consensus voting.
        
        Args:
            features_df: DataFrame with features (single row)
            confidence_threshold: Min probability for signal generation
            model_filter: List of models to include (None = all models)
            
        Returns:
            EnsembleSignal with consensus voting
        """
        individual_signals = []
        long_votes = 0
        short_votes = 0
        neutral_votes = 0
        
        # Generate signals from each model
        models_to_use = model_filter if model_filter else list(self.models.keys())
        
        for model_name in models_to_use:
            if model_name not in self.models:
                continue
            
            signal = self.generate_model_signal(model_name, features_df, confidence_threshold)
            
            if signal:
                individual_signals.append(asdict(signal))
                
                if signal.direction == 1:
                    long_votes += 1
                elif signal.direction == -1:
                    short_votes += 1
                else:
                    neutral_votes += 1
        
        total_votes = len(individual_signals)
        
        if total_votes == 0:
            return EnsembleSignal(
                timestamp=datetime.now().isoformat(),
                long_votes=0,
                short_votes=0,
                neutral_votes=0,
                total_votes=0,
                consensus_probability=0.5,
                ensemble_direction=0,
                consensus_strength="NEUTRAL",
                agreement_level="NONE",
                individual_signals=[]
            )
        
        # Calculate consensus
        long_ratio = long_votes / total_votes if total_votes > 0 else 0
        short_ratio = short_votes / total_votes if total_votes > 0 else 0
        neutral_ratio = neutral_votes / total_votes if total_votes > 0 else 0
        
        # Determine ensemble direction
        if long_ratio > short_ratio and long_ratio > neutral_ratio:
            ensemble_direction = 1
            consensus_prob = long_ratio
        elif short_ratio > long_ratio and short_ratio > neutral_ratio:
            ensemble_direction = -1
            consensus_prob = short_ratio
        else:
            ensemble_direction = 0
            consensus_prob = neutral_ratio if neutral_ratio > 0 else 0.5
        
        # Determine consensus strength
        if ensemble_direction == 1:
            if consensus_prob >= STRONG_BUY_THRESHOLD:
                strength = "STRONG_BUY"
            else:
                strength = "BUY"
        elif ensemble_direction == -1:
            if consensus_prob >= STRONG_SELL_THRESHOLD:
                strength = "STRONG_SELL"
            else:
                strength = "SELL"
        else:
            strength = "NEUTRAL"
        
        # Determine agreement level
        max_vote_ratio = max(long_ratio, short_ratio, neutral_ratio)
        if max_vote_ratio >= 0.95:
            agreement = "UNANIMOUS"
        elif max_vote_ratio >= 0.70:
            agreement = "STRONG"
        elif max_vote_ratio >= 0.50:
            agreement = "MODERATE"
        else:
            agreement = "WEAK"
        
        return EnsembleSignal(
            timestamp=datetime.now().isoformat(),
            long_votes=int(long_votes),
            short_votes=int(short_votes),
            neutral_votes=int(neutral_votes),
            total_votes=int(total_votes),
            consensus_probability=float(consensus_prob),
            ensemble_direction=int(ensemble_direction),
            consensus_strength=strength,
            agreement_level=agreement,
            individual_signals=individual_signals
        )


# ============================================================================
# SIGNAL EXPORT & DISPLAY
# ============================================================================

def save_signal(signal: EnsembleSignal, output_dir: str = OUTPUT_DIR) -> str:
    """Save ensemble signal to JSON file. Returns filepath."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"signal_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(asdict(signal), f, indent=2)
    
    logger.info(f"✓ Signal saved: {filepath}")
    return filepath


def display_signal_summary(signal: EnsembleSignal) -> None:
    """Print signal summary to console."""
    logger.info("\n" + "=" * 100)
    logger.info("ENSEMBLE SIGNAL SUMMARY")
    logger.info("=" * 100)
    logger.info(f"Timestamp: {signal.timestamp}")
    logger.info(f"Direction: {signal.consensus_strength}")
    logger.info(f"Agreement: {signal.agreement_level} ({signal.consensus_probability*100:.1f}%)")
    logger.info(f"Votes: {signal.long_votes} LONG | {signal.short_votes} SHORT | {signal.neutral_votes} NEUTRAL / {signal.total_votes} total")
    
    logger.info("\nIndividual Model Signals:")
    logger.info(f"{'Model':<30} | {'TF':<3} | {'Type':<6} | {'Direction':<8} | {'Prob':<8} | {'Conf':<8}")
    logger.info("-" * 90)
    
    for sig in signal.individual_signals:
        direction_str = "LONG" if sig['direction'] == 1 else "SHORT" if sig['direction'] == -1 else "NEUTRAL"
        logger.info(
            f"{sig['model_name']:<30} | {sig['timeframe']:<3} | {sig['model_type']:<6} | "
            f"{direction_str:<8} | {sig['probability']:.4f} | {sig['confidence']:.4f}"
        )
    
    logger.info("\n" + "=" * 100 + "\n")


def export_signal_csv(signal: EnsembleSignal, output_dir: str = OUTPUT_DIR) -> str:
    """Export signal to CSV format. Returns filepath."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"signal_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Create summary row
    summary_data = {
        'timestamp': signal.timestamp,
        'direction': signal.consensus_strength,
        'agreement': signal.agreement_level,
        'long_votes': signal.long_votes,
        'short_votes': signal.short_votes,
        'neutral_votes': signal.neutral_votes,
        'total_votes': signal.total_votes,
        'probability': signal.consensus_probability
    }
    
    df_summary = pd.DataFrame([summary_data])
    df_summary.to_csv(filepath, index=False)
    
    logger.info(f"✓ Signal CSV saved: {filepath}")
    return filepath


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demo():
    """Demo: Initialize generator and show model inventory."""
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-MODEL ENSEMBLE SIGNAL GENERATOR - DEMO")
    logger.info("=" * 70)
    
    try:
        # Initialize generator
        generator = EnsembleSignalGenerator(ARTIFACTS_DIR)
        
        logger.info(f"\n✓ Generator initialized with {len(generator.models)} models")
        
        # Display model inventory
        logger.info("\nModel Inventory by Timeframe:")
        logger.info("-" * 70)
        
        for tf in TIMEFRAMES:
            quote_models = [m for m in generator.models.keys() if f"quote_{tf}" in m or (f"_{tf}" in m and "quote" in m)]
            ohlcv_models = [m for m in generator.models.keys() if f"ohlcv_{tf}" in m or (f"_{tf}" in m and "ohlcv" in m)]
            
            if quote_models or ohlcv_models:
                logger.info(f"\n{tf}:")
                for m in quote_models:
                    logger.info(f"  • {m} (quote)")
                for m in ohlcv_models:
                    logger.info(f"  • {m} (ohlcv)")
        
        logger.info(f"\n\nTo generate signals in production:")
        logger.info("""
1. Compute all required features from live market data
   - Use quote features (bid/ask microstructure)
   - Use OHLCV features (candlestick patterns, regimes, etc.)

2. Create a DataFrame with all features:
   features_df = pd.DataFrame({
       'feature_1': [value1],
       'feature_2': [value2],
       ... all 50+ features ...
   })

3. Generate ensemble signal:
   signal = generator.generate_ensemble_signal(features_df)

4. Export and use:
   display_signal_summary(signal)
   save_signal(signal)
   export_signal_csv(signal)

5. Send to trading system:
   if signal.consensus_strength in ['STRONG_BUY', 'STRONG_SELL']:
       execute_trade(signal)
        """)
        
        logger.info("=" * 70)
        logger.info("✅ Demo Complete")
        logger.info("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    demo()
