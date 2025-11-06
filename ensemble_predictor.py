#!/usr/bin/env python3
"""
MULTI-MODEL ENSEMBLE PREDICTOR
===============================
Combines predictions from multiple models (across different timeframes)
for improved accuracy and confidence.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Import model class for unpickling
sys.path.insert(0, str(Path(__file__).parent))
from production_final_system import BalancedModel


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models for a symbol.
    
    Strategies:
    1. Majority vote: Simple democracy (each model gets 1 vote)
    2. Confidence-weighted: Weight by prediction confidence
    3. Performance-weighted: Weight by recent win rate from backtest
    """
    
    def __init__(self, symbol: str, model_dir: Path = Path("models_production")):
        """
        Initialize ensemble for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            model_dir: Directory containing production models
        """
        self.symbol = symbol
        self.model_dir = model_dir
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all production-ready models for this symbol."""
        symbol_dir = self.model_dir / self.symbol
        
        if not symbol_dir.exists():
            print(f"  ⚠️  No models found for {self.symbol}")
            return
        
        # Load all PRODUCTION_READY models
        for model_file in symbol_dir.glob("*_PRODUCTION_READY.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract timeframe from filename (e.g., "XAUUSD_5T_PRODUCTION_READY.pkl" -> "5T")
                parts = model_file.stem.split('_')
                timeframe = parts[1]  # Assuming format SYMBOL_TF_STATUS
                
                self.models[timeframe] = {
                    'model': data['model'],
                    'features': data['features'],
                    'results': data.get('results', {}),
                    'params': data.get('params', {})
                }
                
            except Exception as e:
                print(f"  ⚠️  Failed to load {model_file.name}: {e}")
        
        print(f"  ✅ Loaded {len(self.models)} models for {self.symbol}: {list(self.models.keys())}")
    
    def predict_single(self, timeframe: str, features: pd.DataFrame) -> Optional[Dict]:
        """
        Get prediction from a single model.
        
        Args:
            timeframe: Model timeframe (e.g., '5T')
            features: Feature DataFrame (must contain all required features)
        
        Returns:
            Dict with prediction, probabilities, confidence, edge
        """
        if timeframe not in self.models:
            return None
        
        model_data = self.models[timeframe]
        model = model_data['model']
        required_features = model_data['features']
        
        # Ensure all required features are present
        missing_features = set(required_features) - set(features.columns)
        if missing_features:
            print(f"  ⚠️  Missing features for {timeframe}: {missing_features}")
            return None
        
        # Extract features in correct order and predict
        X = features[required_features].fillna(0).values
        
        try:
            probs = model.predict_proba(X)
            
            # probs shape: (n_samples, 3) for [Flat, Up, Down]
            if len(probs.shape) == 2:
                probs = probs[-1]  # Get last row if multiple samples
            
            flat_prob, up_prob, down_prob = probs[0], probs[1], probs[2]
            
            # Determine prediction
            max_prob = max(flat_prob, up_prob, down_prob)
            
            if up_prob == max_prob:
                prediction = 'long'
            elif down_prob == max_prob:
                prediction = 'short'
            else:
                prediction = 'flat'
            
            # Calculate edge (difference between highest and second-highest probability)
            sorted_probs = sorted([flat_prob, up_prob, down_prob], reverse=True)
            edge = sorted_probs[0] - sorted_probs[1]
            
            # Get win rate from backtest results
            win_rate = model_data['results'].get('win_rate', 50.0) / 100.0
            
            return {
                'timeframe': timeframe,
                'prediction': prediction,
                'probabilities': {'flat': flat_prob, 'up': up_prob, 'down': down_prob},
                'confidence': max_prob,
                'edge': edge,
                'win_rate': win_rate
            }
            
        except Exception as e:
            print(f"  ❌ Prediction error for {timeframe}: {e}")
            return None
    
    def predict_ensemble(self, features: pd.DataFrame, 
                        strategy: str = 'performance_weighted',
                        min_models: int = 2) -> Optional[Dict]:
        """
        Generate ensemble prediction from all available models.
        
        Args:
            features: Feature DataFrame
            strategy: Voting strategy ('majority', 'confidence_weighted', 'performance_weighted')
            min_models: Minimum number of models required for ensemble
        
        Returns:
            Dict with ensemble prediction, confidence, and voting details
        """
        if len(self.models) < min_models:
            print(f"  ⚠️  Only {len(self.models)} models available (need {min_models})")
            return None
        
        # Get predictions from all models
        predictions = []
        for timeframe in self.models.keys():
            pred = self.predict_single(timeframe, features)
            if pred and pred['prediction'] != 'flat':  # Only use directional predictions
                predictions.append(pred)
        
        if len(predictions) < min_models:
            print(f"  ⚠️  Only {len(predictions)} valid predictions (need {min_models})")
            return None
        
        # Apply voting strategy
        if strategy == 'majority':
            return self._majority_vote(predictions)
        elif strategy == 'confidence_weighted':
            return self._confidence_weighted_vote(predictions)
        elif strategy == 'performance_weighted':
            return self._performance_weighted_vote(predictions)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _majority_vote(self, predictions: List[Dict]) -> Dict:
        """Simple majority voting (each model gets 1 vote)."""
        votes = {'long': 0, 'short': 0}
        
        for pred in predictions:
            if pred['prediction'] in votes:
                votes[pred['prediction']] += 1
        
        # Determine winner
        if votes['long'] > votes['short']:
            ensemble_pred = 'long'
        elif votes['short'] > votes['long']:
            ensemble_pred = 'short'
        else:
            # Tie - use confidence as tiebreaker
            long_confidence = np.mean([p['confidence'] for p in predictions if p['prediction'] == 'long'])
            short_confidence = np.mean([p['confidence'] for p in predictions if p['prediction'] == 'short'])
            ensemble_pred = 'long' if long_confidence > short_confidence else 'short'
        
        # Calculate ensemble confidence (average of agreeing models)
        agreeing = [p for p in predictions if p['prediction'] == ensemble_pred]
        ensemble_confidence = np.mean([p['confidence'] for p in agreeing])
        ensemble_edge = np.mean([p['edge'] for p in agreeing])
        
        return {
            'prediction': ensemble_pred,
            'confidence': ensemble_confidence,
            'edge': ensemble_edge,
            'strategy': 'majority',
            'votes': votes,
            'num_models': len(predictions),
            'agreeing_models': len(agreeing),
            'details': predictions
        }
    
    def _confidence_weighted_vote(self, predictions: List[Dict]) -> Dict:
        """Confidence-weighted voting (models with higher confidence get more weight)."""
        weighted_votes = {'long': 0.0, 'short': 0.0}
        
        for pred in predictions:
            if pred['prediction'] in weighted_votes:
                weighted_votes[pred['prediction']] += pred['confidence']
        
        # Determine winner
        ensemble_pred = 'long' if weighted_votes['long'] > weighted_votes['short'] else 'short'
        
        # Calculate ensemble confidence
        total_weight = sum(weighted_votes.values())
        ensemble_confidence = weighted_votes[ensemble_pred] / total_weight if total_weight > 0 else 0.5
        
        # Calculate ensemble edge
        agreeing = [p for p in predictions if p['prediction'] == ensemble_pred]
        ensemble_edge = np.mean([p['edge'] for p in agreeing]) if agreeing else 0.0
        
        return {
            'prediction': ensemble_pred,
            'confidence': ensemble_confidence,
            'edge': ensemble_edge,
            'strategy': 'confidence_weighted',
            'weighted_votes': weighted_votes,
            'num_models': len(predictions),
            'agreeing_models': len(agreeing),
            'details': predictions
        }
    
    def _performance_weighted_vote(self, predictions: List[Dict]) -> Dict:
        """Performance-weighted voting (models with higher win rates get more weight)."""
        weighted_votes = {'long': 0.0, 'short': 0.0}
        
        for pred in predictions:
            if pred['prediction'] in weighted_votes:
                # Weight by win_rate (0.5 to 1.0 typically)
                weight = pred['win_rate']
                weighted_votes[pred['prediction']] += weight
        
        # Determine winner
        ensemble_pred = 'long' if weighted_votes['long'] > weighted_votes['short'] else 'short'
        
        # Calculate ensemble metrics
        agreeing = [p for p in predictions if p['prediction'] == ensemble_pred]
        
        # Weight confidence by performance
        if agreeing:
            total_weight = sum(p['win_rate'] for p in agreeing)
            ensemble_confidence = sum(p['confidence'] * p['win_rate'] for p in agreeing) / total_weight
            ensemble_edge = sum(p['edge'] * p['win_rate'] for p in agreeing) / total_weight
        else:
            ensemble_confidence = 0.5
            ensemble_edge = 0.0
        
        return {
            'prediction': ensemble_pred,
            'confidence': ensemble_confidence,
            'edge': ensemble_edge,
            'strategy': 'performance_weighted',
            'weighted_votes': weighted_votes,
            'num_models': len(predictions),
            'agreeing_models': len(agreeing),
            'details': predictions
        }
    
    def get_model_count(self) -> int:
        """Return number of loaded models."""
        return len(self.models)
    
    def get_timeframes(self) -> List[str]:
        """Return list of available timeframes."""
        return list(self.models.keys())


def test_ensemble():
    """Test ensemble predictor."""
    import pandas as pd
    
    print("\n" + "="*80)
    print("TESTING ENSEMBLE PREDICTOR")
    print("="*80 + "\n")
    
    # Test with XAUUSD
    ensemble = EnsemblePredictor('XAUUSD')
    
    if ensemble.get_model_count() == 0:
        print("❌ No models loaded. Exiting.")
        return
    
    # Create dummy features (in real use, these come from calculate_features())
    # For testing, we'll use the actual feature names from a loaded model
    sample_model = list(ensemble.models.values())[0]
    feature_names = sample_model['features']
    
    # Generate random features for testing
    features_df = pd.DataFrame(
        np.random.randn(1, len(feature_names)),
        columns=feature_names
    )
    
    print(f"Testing with {len(feature_names)} features\n")
    
    # Test all strategies
    for strategy in ['majority', 'confidence_weighted', 'performance_weighted']:
        print(f"\n--- Strategy: {strategy} ---")
        result = ensemble.predict_ensemble(features_df, strategy=strategy)
        
        if result:
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Edge: {result['edge']:.3f}")
            print(f"Models: {result['num_models']} total, {result['agreeing_models']} agreeing")
            print(f"Votes: {result.get('votes', result.get('weighted_votes', {}))}")
        else:
            print("❌ Prediction failed")
    
    print("\n" + "="*80)
    print("✅ Test complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_ensemble()

