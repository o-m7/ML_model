"""
Quote-based ML Trading Models
Training system using bid/ask quote features and order flow signals
Similar methodology to citadel_v6.py but using quote data instead of OHLCV
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class QuoteModelTrainer:
    """Train ML models on quote/orderflow features"""
    
    def __init__(self, symbol='C:XAU-USD', lookback=20, target_periods=1):
        self.symbol = symbol
        self.lookback = lookback  # Feature window
        self.target_periods = target_periods  # Periods ahead for label
        self.models_dir = Path('QUOTE_models')
        self.data_dir = Path('feature_store') / symbol / 'quotes'
        
        self.models_dir.mkdir(exist_ok=True)
        
    def load_quote_data(self, timeframe='5T'):
        """Load quote data with features"""
        path = self.data_dir / f'{self.symbol}_{timeframe}_quotes_with_features.parquet'
        df = pd.read_parquet(path)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} bars from {timeframe}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def create_labels(self, df, target_periods=1):
        """Create future return labels (1=up, 0=down)"""
        df['close_fwd'] = df['close'].shift(-target_periods)
        df['return_fwd'] = (df['close_fwd'] - df['close']) / df['close']
        df['label'] = (df['return_fwd'] > 0).astype(int)
        
        return df
    
    def create_feature_windows(self, df):
        """Create rolling feature windows (no look-ahead)"""
        feature_cols = [
            # Quote core
            'open', 'high', 'low', 'close', 'volume',
            # Spread dynamics
            'spread', 'spread_pct', 'spread_volatility', 'spread_momentum',
            'spread_mean_reversion',
            # Bid-ask imbalance
            'bid_ask_imbalance', 'bid_ask_imbalance_ema',
            'cumulative_imbalance', 'imbalance_volatility', 'imbalance_skew',
            # Order flow toxicity
            'order_flow_toxicity', 'adverse_selection', 'information_leakage',
            'toxicity_trend',
            # Depth analysis
            'depth_imbalance', 'depth_ratio',
            # Quote volatility
            'quote_intensity', 'quote_volatility', 'quote_volatility_ema',
            # Microstructure
            'quote_clustering', 'bid_ask_coherence',
            'price_efficiency', 'effective_spread', 'realized_spread'
        ]
        
        # Filter to available columns
        available = [c for c in feature_cols if c in df.columns]
        logger.info(f"Using {len(available)} features: {available[:5]}...")
        
        # Create lagged features (no look-ahead)
        X = pd.DataFrame(index=df.index)
        for lag in range(1, self.lookback + 1):
            for col in available:
                X[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # Drop NaN rows from lagging
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = df.loc[valid_idx, 'label']
        
        logger.info(f"Created feature matrix: {X.shape}")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def walk_forward_train(self, df, strategy_name, timeframe, initial_train_size=5000, step_size=500):
        """Walk-forward validation training"""
        logger.info(f"\n{'='*70}")
        logger.info(f"WALK-FORWARD: {strategy_name.upper()} | {timeframe}")
        logger.info(f"{'='*70}")
        
        # Prepare data
        df = self.create_labels(df, target_periods=self.target_periods)
        X, y = self.create_feature_windows(df)
        
        if len(X) < initial_train_size + step_size:
            logger.warning(f"Insufficient data: {len(X)} < {initial_train_size + step_size}")
            return None
        
        # Walk-forward loop
        models = []
        predictions = []
        actuals = []
        
        for step in range(0, len(X) - initial_train_size - step_size, step_size):
            train_end = initial_train_size + step
            test_start = train_end
            test_end = test_start + step_size
            
            if test_end > len(X):
                test_end = len(X)
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost
            if strategy_name == 'mean_reversion':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                )
            elif strategy_name == 'volatility_breakout':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    random_state=42,
                    eval_metric='logloss'
                )
            else:  # trend_following
                model = xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    random_state=42,
                    eval_metric='logloss'
                )
            
            model.fit(X_train_scaled, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            models.append(model)
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            
            acc = (y_pred == y_test).mean()
            logger.info(f"  Window {step//step_size + 1}: Accuracy={acc:.2%}")
        
        # Overall metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        overall_acc = (predictions == actuals).mean()
        win_rate = (predictions == 1) & (actuals == 1)
        win_rate = win_rate.sum() / (predictions == 1).sum() if (predictions == 1).sum() > 0 else 0
        
        logger.info(f"\nOverall Accuracy: {overall_acc:.2%}")
        logger.info(f"Win Rate (Precision): {win_rate:.2%}")
        logger.info(f"Signal Distribution: {(predictions == 1).sum()} ups / {(predictions == 0).sum()} downs")
        
        # Use last model for production
        final_model = models[-1]
        
        return final_model
    
    def train_all_strategies(self):
        """Train all three quote-based strategies"""
        all_models = {}
        
        for timeframe in ['5T', '15T']:
            logger.info(f"\n{'#'*70}")
            logger.info(f"TIMEFRAME: {timeframe}")
            logger.info(f"{'#'*70}")
            
            df = self.load_quote_data(timeframe)
            
            for strategy in ['mean_reversion', 'volatility_breakout', 'trend_following']:
                model = self.walk_forward_train(df, strategy, timeframe)
                
                if model:
                    key = f'{strategy}_{self.symbol}_{timeframe}'
                    all_models[key] = model
                    
                    # Save
                    model_path = self.models_dir / f'{strategy}_{self.symbol}_{timeframe}.pkl'
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.info(f"✓ Saved: {model_path}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ QUOTE MODEL TRAINING COMPLETE")
        logger.info(f"Trained {len(all_models)} models in QUOTE_models/")
        logger.info(f"{'='*70}")
        
        return all_models


def main():
    trainer = QuoteModelTrainer(symbol='C:XAU-USD', lookback=20, target_periods=1)
    trainer.train_all_strategies()


if __name__ == '__main__':
    main()
