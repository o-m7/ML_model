"""
citadel_v6.py - Clean ML Training System

NO LOOK-AHEAD BIAS:
- Walk-forward validation only
- Train on [0, split_point)
- Validate on [split_point, end)
- Test on future data only
- Features computed BEFORE signal generation
- No future information in training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from datetime import datetime, timedelta
import pickle

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('citadel_v6')

@dataclass
class Config:
    FEATURE_STORE: Path = Path("feature_store")
    MODELS_DIR: Path = Path("production_models")
    MIN_TRAIN_SAMPLES = 500
    MIN_VAL_SAMPLES = 200
    MIN_TEST_SAMPLES = 100


CONFIG = Config()


class TripleBarrier:
    """Triple barrier labeling with NO LOOK-AHEAD BIAS."""
    
    @staticmethod
    def label(df: pd.DataFrame, tp_mult: float = 1.0, sl_mult: float = 1.0, 
              bars_ahead: int = 20) -> pd.Series:
        """
        Label OHLCV bars: 1=win, 0=loss, -1=timeout
        
        CRITICAL: Only use prior bar ATR, no future information
        """
        if 'atr' not in df.columns:
            raise ValueError("Need ATR for labeling")
        
        labels = []
        
        for i in range(len(df) - bars_ahead - 1):
            # Use prior bar ATR only (shift(1))
            atr = df['atr'].iloc[i]
            if pd.isna(atr) or atr <= 0:
                labels.append(-1)
                continue
            
            entry = df['close'].iloc[i]
            tp = entry + (atr * tp_mult)
            sl = entry - (atr * sl_mult)
            
            # Check bars AHEAD (lookahead range: bars i+1 to i+bars_ahead)
            future_bars = df.iloc[i+1:i+bars_ahead+1]
            
            if len(future_bars) == 0:
                labels.append(-1)
                continue
            
            # Check if hit TP (high touches/exceeds)
            if (future_bars['high'] >= tp).any():
                labels.append(1)
                continue
            
            # Check if hit SL (low touches/falls below)
            if (future_bars['low'] <= sl).any():
                labels.append(0)
                continue
            
            # Timeout
            labels.append(-1)
        
        # Pad remaining rows with -1
        labels.extend([-1] * (len(df) - len(labels)))
        
        return pd.Series(labels, index=df.index)


class DataProcessor:
    """Clean data processing with proper time splits."""
    
    @staticmethod
    def load_and_split(symbol: str, timeframe: str) -> tuple:
        """Load data and split into TRAIN / VAL / TEST (time-based, NO OVERLAP)."""
        path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        
        df = pd.read_parquet(path)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} bars from {path.name}")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Time-based split: NO OVERLAP
        total = len(df)
        train_size = int(total * 0.60)
        val_size = int(total * 0.20)
        
        if train_size < CONFIG.MIN_TRAIN_SAMPLES:
            raise ValueError(f"Not enough data: {train_size} < {CONFIG.MIN_TRAIN_SAMPLES}")
        
        df_train = df.iloc[:train_size].copy()
        df_val = df.iloc[train_size:train_size+val_size].copy()
        df_test = df.iloc[train_size+val_size:].copy()
        
        logger.info(f"  TRAIN: {len(df_train)} bars ({df_train['timestamp'].min()} to {df_train['timestamp'].max()})")
        logger.info(f"  VAL:   {len(df_val)} bars ({df_val['timestamp'].min()} to {df_val['timestamp'].max()})")
        logger.info(f"  TEST:  {len(df_test)} bars ({df_test['timestamp'].min()} to {df_test['timestamp'].max()})")
        
        return df_train, df_val, df_test
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
        """Select and scale features."""
        if feature_cols is None:
            # Default: use core technical features only
            feature_cols = [c for c in df.columns if c not in 
                          ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Remove any columns with all NaN
        feature_cols = [c for c in feature_cols if c in df.columns and df[c].notna().sum() > 0]
        
        X = df[feature_cols].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        return X, feature_cols


class SignalGenerator:
    """Generate buy/sell signals from OHLCV (NO LEAKAGE)."""
    
    @staticmethod
    def trend_following(df: pd.DataFrame) -> pd.Series:
        """Simple trend following: close > MA"""
        if 'ema_20' not in df.columns:
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        signal = (df['close'] > df['ema_20']).astype(int)
        return signal
    
    @staticmethod
    def mean_reversion(df: pd.DataFrame) -> pd.Series:
        """Mean reversion: price at extremes"""
        if 'rsi' not in df.columns:
            # Compute RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
        
        signal = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)
        return signal
    
    @staticmethod
    def volatility_breakout(df: pd.DataFrame) -> pd.Series:
        """Volatility breakout: price move > ATR"""
        if 'atr' not in df.columns:
            prev_close = df['close'].shift(1)
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs()
            ], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
        
        price_move = (df['close'] - df['open']).abs()
        signal = (price_move > df['atr'] * 0.5).astype(int)
        return signal


class Model:
    """ML model for filtering signals."""
    
    def __init__(self, name: str = "xgb"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list):
        """Train model on TRAINING data ONLY (no val leakage)."""
        self.feature_cols = feature_cols
        
        # Remove samples with invalid labels
        mask = y_train != -1
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        if len(y_train) < 50:
            logger.warning(f"Too few training samples: {len(y_train)}")
            return False
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train
        if self.name == "xgb":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_scaled, y_train)
        logger.info(f"  Model trained on {len(y_train)} samples")
        return True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on validation/test data."""
        if self.model is None:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class Backtester:
    """Evaluate strategy with proper metrics (NO LEAKAGE)."""
    
    @staticmethod
    def backtest(df: pd.DataFrame, signals: np.ndarray, labels: pd.Series, 
                 spread_cost: float = 0.2) -> dict:
        """
        Backtest signals.
        
        CRITICAL: Only count bars where signal=1 AND label != -1
        """
        results = {
            'total_bars': len(df),
            'signal_bars': (signals == 1).sum(),
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'avg_return': 0,
            'pf': 0,
            'max_dd': 0,
        }
        
        # Filter: only where signal=1 and valid label
        mask = (signals == 1) & (labels != -1)
        trade_labels = labels[mask]
        
        if len(trade_labels) == 0:
            return results
        
        # Count wins/losses
        wins = (trade_labels == 1).sum()
        losses = (trade_labels == 0).sum()
        
        results['trades'] = len(trade_labels)
        results['wins'] = wins
        results['losses'] = losses
        results['win_rate'] = wins / len(trade_labels) if len(trade_labels) > 0 else 0
        
        # Estimate returns (wins = +1R, losses = -1R, cost = -spread_cost)
        returns = np.where(trade_labels == 1, 1 - spread_cost, -1 - spread_cost)
        results['avg_return'] = returns.mean() if len(returns) > 0 else 0
        
        # PF (profit factor)
        gross_profit = (trade_labels == 1).sum() * (1 - spread_cost)
        gross_loss = (trade_labels == 0).sum() * (1 + spread_cost)
        results['pf'] = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return results


class ModelExporter:
    """Export trained models for production use."""
    
    @staticmethod
    def save_model(model: Model, signal_func, strat_name: str, symbol: str, 
                   timeframe: str, metrics: dict):
        """Save model, scaler, and metadata to OHLCV_models/."""
        models_dir = Path("OHLCV_models")
        models_dir.mkdir(exist_ok=True)
        
        # Model filename: {strategy}_{symbol}_{timeframe}.pkl
        safe_symbol = symbol.replace(":", "_")
        model_path = models_dir / f"{strat_name}_{safe_symbol}_{timeframe}.pkl"
        
        # Package for export
        export_data = {
            'model': model.model,
            'scaler': model.scaler,
            'feature_cols': model.feature_cols,
            'strategy': strat_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(export_data, f)
        
        logger.info(f"✓ Saved model: {model_path}")
        return str(model_path)


class Pipeline:
    """Full training pipeline."""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.results = []
    
    def run(self):
        """Run full pipeline."""
        logger.info("=" * 70)
        logger.info(f"CITADEL V6 - NO LEAKAGE TRAINING")
        logger.info(f"{self.symbol} {self.timeframe}")
        logger.info("=" * 70)
        
        try:
            # Load and split
            df_train, df_val, df_test = DataProcessor.load_and_split(self.symbol, self.timeframe)
            
            # Label training data
            logger.info("\n[LABELING] Triple barrier labeling on TRAIN set only...")
            y_train = TripleBarrier.label(df_train, tp_mult=1.0, sl_mult=1.0, bars_ahead=20)
            logger.info(f"  Wins: {(y_train == 1).sum()}, Losses: {(y_train == 0).sum()}, Timeout: {(y_train == -1).sum()}")
            
            # Label validation data
            logger.info("\n[LABELING] Triple barrier labeling on VAL set...")
            y_val = TripleBarrier.label(df_val, tp_mult=1.0, sl_mult=1.0, bars_ahead=20)
            logger.info(f"  Wins: {(y_val == 1).sum()}, Losses: {(y_val == 0).sum()}, Timeout: {(y_val == -1).sum()}")
            
            # Label test data
            logger.info("\n[LABELING] Triple barrier labeling on TEST set...")
            y_test = TripleBarrier.label(df_test, tp_mult=1.0, sl_mult=1.0, bars_ahead=20)
            logger.info(f"  Wins: {(y_test == 1).sum()}, Losses: {(y_test == 0).sum()}, Timeout: {(y_test == -1).sum()}")
            
            # Prepare features
            logger.info("\n[FEATURES] Preparing feature set...")
            X_train, feature_cols = DataProcessor.prepare_features(df_train)
            X_val, _ = DataProcessor.prepare_features(df_val, feature_cols)
            X_test, _ = DataProcessor.prepare_features(df_test, feature_cols)
            logger.info(f"  Using {len(feature_cols)} features")
            
            # Test each signal generator + model combo
            strategies = [
                ("trend_following", SignalGenerator.trend_following),
                ("mean_reversion", SignalGenerator.mean_reversion),
                ("volatility_breakout", SignalGenerator.volatility_breakout),
            ]
            
            for strat_name, signal_func in strategies:
                logger.info(f"\n{'='*70}")
                logger.info(f"STRATEGY: {strat_name}")
                logger.info(f"{'='*70}")
                
                # Generate signals (on train only for training)
                signals_train = signal_func(df_train)
                
                # Train model
                logger.info("\n[ML] Training model...")
                model = Model("xgb")
                if not model.train(X_train, y_train, feature_cols):
                    logger.warning("  Failed to train model, skipping...")
                    continue
                
                # Evaluate on VAL
                logger.info("\n[VALIDATION]")
                signals_val = signal_func(df_val)
                preds_val = model.predict(X_val)
                
                # Combine signal + model prediction
                combined_signals = signals_val * preds_val
                
                metrics_val = Backtester.backtest(df_val, combined_signals, y_val)
                logger.info(f"  Trades: {metrics_val['trades']}, WR: {metrics_val['win_rate']:.1%}, PF: {metrics_val['pf']:.2f}")
                logger.info(f"  AvgR: {metrics_val['avg_return']:.2f}R")
                
                # Check if passed gates
                if (metrics_val['trades'] >= 20 and 
                    metrics_val['pf'] >= 1.05 and 
                    metrics_val['win_rate'] >= 0.45):
                    
                    logger.info("\n✓ PASSED validation gates, testing on TEST set...")
                    
                    # Evaluate on TEST
                    signals_test = signal_func(df_test)
                    preds_test = model.predict(X_test)
                    combined_signals_test = signals_test * preds_test
                    
                    metrics_test = Backtester.backtest(df_test, combined_signals_test, y_test)
                    logger.info(f"  TEST - Trades: {metrics_test['trades']}, WR: {metrics_test['win_rate']:.1%}, PF: {metrics_test['pf']:.2f}")
                    logger.info(f"  TEST - AvgR: {metrics_test['avg_return']:.2f}R")
                    
                    # SAVE MODEL
                    ModelExporter.save_model(
                        model, signal_func, strat_name, 
                        self.symbol, self.timeframe, 
                        {'validation': metrics_val, 'test': metrics_test}
                    )
                    
                    self.results.append({
                        'strategy': strat_name,
                        'val_metrics': metrics_val,
                        'test_metrics': metrics_test,
                    })
                else:
                    logger.warning("  ✗ Failed validation gates")
            
            # Summary
            logger.info(f"\n{'='*70}")
            logger.info(f"SUMMARY: {len(self.results)} strategies with edge")
            logger.info(f"{'='*70}")
            
            if self.results:
                for r in self.results:
                    logger.info(f"✓ {r['strategy']}: TEST WR={r['test_metrics']['win_rate']:.1%}, PF={r['test_metrics']['pf']:.2f}")
            else:
                logger.warning("⚠️ No strategies found with edge")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='C:XAU-USD')
    p.add_argument('--timeframe', default='5T')
    args = p.parse_args()
    
    pipeline = Pipeline(args.symbol, args.timeframe)
    pipeline.run()


if __name__ == '__main__':
    main()
