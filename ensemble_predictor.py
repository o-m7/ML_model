#!/usr/bin/env python3
"""
MULTI-MODEL ENSEMBLE PREDICTOR
================================
Combines predictions from multiple models (different timeframes) for better accuracy.
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Import model class for unpickling
sys.path.insert(0, str(Path(__file__).parent))
from production_final_system import BalancedModel

warnings.filterwarnings('ignore')


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models for a symbol.
    
    Voting strategies:
    1. Majority vote - Simple democracy
    2. Confidence-weighted - Weight by model confidence
    3. Performance-weighted - Weight by recent win rate
    """
    
    def __init__(self, symbol: str, models_dir: Path = Path("models_production")):
        self.symbol = symbol
        self.models_dir = models_dir
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all production-ready models for the symbol."""
        symbol_dir = self.models_dir / self.symbol
        
        if not symbol_dir.exists():
            raise ValueError(f"No models found for {self.symbol}")
        
        # Load all PRODUCTION_READY models
        for model_file in symbol_dir.glob(f"{self.symbol}_*_PRODUCTION_READY.pkl"):
            timeframe = model_file.stem.split('_')[1]  # Extract timeframe from filename
            
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models[timeframe] = {
                        'model': model_data['model'],
                        'features': model_data['features'],
                        'metadata': model_data.get('metadata', {}),
                        'results': model_data.get('results', {})
                    }
                    print(f"  ✅ Loaded {self.symbol} {timeframe} model")
            except Exception as e:
                print(f"  ⚠️  Failed to load {self.symbol} {timeframe}: {e}")
        
        if not self.models:
            raise ValueError(f"No production-ready models found for {self.symbol}")
        
        print(f"  📊 Loaded {len(self.models)} models for {self.symbol}: {list(self.models.keys())}")
    
    def predict_single_model(self, timeframe: str, features: pd.DataFrame) -> Optional[Dict]:
        """Get prediction from a single model."""
        if timeframe not in self.models:
            return None
        
        model_info = self.models[timeframe]
        model = model_info['model']
        required_features = model_info['features']
        
        # Ensure we have all required features
        missing_features = set(required_features) - set(features.columns)
        if missing_features:
            print(f"  ⚠️  {timeframe}: Missing features {missing_features}")
            return None
        
        # Prepare features in correct order
        X = features[required_features].fillna(0).values
        
        # Get prediction
        try:
            probs = model.predict_proba(X)
            
            # probs shape: (n_samples, n_classes) where classes are [Flat, Up, Down]
            flat_prob = float(probs[0, 0])
            long_prob = float(probs[0, 1])
            short_prob = float(probs[0, 2])
            
            # Determine signal
            max_prob = max(long_prob, short_prob)
            
            if long_prob > short_prob and long_prob > flat_prob:
                signal = 'long'
                confidence = long_prob
            elif short_prob > long_prob and short_prob > flat_prob:
                signal = 'short'
                confidence = short_prob
            else:
                signal = 'flat'
                confidence = flat_prob
            
            # Calculate edge (difference between top 2 probabilities)
            sorted_probs = sorted([flat_prob, long_prob, short_prob], reverse=True)
            edge = sorted_probs[0] - sorted_probs[1]
            
            return {
                'timeframe': timeframe,
                'signal': signal,
                'confidence': confidence,
                'edge': edge,
                'probs': {'flat': flat_prob, 'long': long_prob, 'short': short_prob}
            }
        
        except Exception as e:
            print(f"  ❌ {timeframe}: Prediction error: {e}")
            return None
    
    def ensemble_predict(self, features: pd.DataFrame, strategy: str = 'performance_weighted') -> Dict:
        """
        Get ensemble prediction from all available models.
        
        Args:
            features: DataFrame with calculated features (single row)
            strategy: 'majority', 'confidence_weighted', or 'performance_weighted'
        
        Returns:
            Dict with ensemble prediction details
        """
        predictions = []
        
        # Get predictions from all models
        for timeframe in self.models.keys():
            pred = self.predict_single_model(timeframe, features)
            if pred:
                predictions.append(pred)
        
        if not predictions:
            return {
                'signal': 'flat',
                'confidence': 0.0,
                'edge': 0.0,
                'num_models': 0,
                'votes': {}
            }
        
        # Apply voting strategy
        if strategy == 'majority':
            ensemble_result = self._majority_vote(predictions)
        elif strategy == 'confidence_weighted':
            ensemble_result = self._confidence_weighted_vote(predictions)
        elif strategy == 'performance_weighted':
            ensemble_result = self._performance_weighted_vote(predictions)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        ensemble_result['num_models'] = len(predictions)
        ensemble_result['individual_predictions'] = predictions
        
        return ensemble_result
    
    def _majority_vote(self, predictions: List[Dict]) -> Dict:
        """Simple majority voting."""
        votes = {'long': 0, 'short': 0, 'flat': 0}
        total_confidence = 0
        total_edge = 0
        
        for pred in predictions:
            votes[pred['signal']] += 1
            total_confidence += pred['confidence']
            total_edge += pred['edge']
        
        # Determine winner
        winner = max(votes, key=votes.get)
        
        return {
            'signal': winner,
            'confidence': total_confidence / len(predictions),
            'edge': total_edge / len(predictions),
            'votes': votes,
            'strategy': 'majority'
        }
    
    def _confidence_weighted_vote(self, predictions: List[Dict]) -> Dict:
        """Vote weighted by model confidence."""
        weighted_scores = {'long': 0.0, 'short': 0.0, 'flat': 0.0}
        total_weight = 0.0
        
        for pred in predictions:
            weight = pred['confidence']
            weighted_scores[pred['signal']] += weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for signal in weighted_scores:
                weighted_scores[signal] /= total_weight
        
        # Determine winner
        winner = max(weighted_scores, key=weighted_scores.get)
        confidence = weighted_scores[winner]
        
        # Calculate edge
        sorted_scores = sorted(weighted_scores.values(), reverse=True)
        edge = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        
        return {
            'signal': winner,
            'confidence': confidence,
            'edge': edge,
            'votes': {k: f"{v:.3f}" for k, v in weighted_scores.items()},
            'strategy': 'confidence_weighted'
        }
    
    def _performance_weighted_vote(self, predictions: List[Dict]) -> Dict:
        """Vote weighted by recent model performance (win rate)."""
        weighted_scores = {'long': 0.0, 'short': 0.0, 'flat': 0.0}
        total_weight = 0.0
        
        for pred in predictions:
            timeframe = pred['timeframe']
            model_info = self.models.get(timeframe, {})
            results = model_info.get('results', {})
            
            # Use win rate as weight (default to 0.5 if not available)
            win_rate = results.get('win_rate', 50.0) / 100.0
            
            # Combine win rate with confidence for weight
            weight = win_rate * pred['confidence']
            
            weighted_scores[pred['signal']] += weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for signal in weighted_scores:
                weighted_scores[signal] /= total_weight
        
        # Determine winner
        winner = max(weighted_scores, key=weighted_scores.get)
        confidence = weighted_scores[winner]
        
        # Calculate edge
        sorted_scores = sorted(weighted_scores.values(), reverse=True)
        edge = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        
        return {
            'signal': winner,
            'confidence': confidence,
            'edge': edge,
            'votes': {k: f"{v:.3f}" for k, v in weighted_scores.items()},
            'strategy': 'performance_weighted'
        }


def test_ensemble():
    """Test the ensemble predictor."""
    print("\n" + "="*80)
    print("TESTING ENSEMBLE PREDICTOR")
    print("="*80 + "\n")
    
    # Test with XAUUSD
    try:
        ensemble = EnsemblePredictor('XAUUSD')
        print(f"\n✅ Successfully loaded ensemble for XAUUSD with {len(ensemble.models)} models")
        print(f"   Timeframes: {list(ensemble.models.keys())}")
    except Exception as e:
        print(f"❌ Failed to load ensemble: {e}")


if __name__ == "__main__":
    test_ensemble()
