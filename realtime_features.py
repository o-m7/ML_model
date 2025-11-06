#!/usr/bin/env python3
"""
Real-time Feature Computation Engine
Computes all technical indicators from recent bars for live signal generation.

This solves the problem: you can't compute indicators from just 1 bar,
you need recent historical data.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

class RealtimeFeatureEngine:
    """
    Compute features in real-time from recent historical bars.
    Maintains consistency with training feature engineering.
    """
    
    def __init__(self, lookback_bars: int = 200):
        """
        Args:
            lookback_bars: Number of recent bars needed to compute indicators
        """
        self.lookback_bars = lookback_bars
    
    def compute_features(self, recent_bars: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Compute all features from recent bars.
        
        Args:
            recent_bars: DataFrame with columns [timestamp, open, high, low, close, volume]
                        Must have at least lookback_bars rows
        
        Returns:
            latest_features: Dict of feature values for the latest bar
            df_with_features: Full DataFrame with all computed features
        """
        if len(recent_bars) < self.lookback_bars:
            raise ValueError(f"Need at least {self.lookback_bars} bars, got {len(recent_bars)}")
        
        df = recent_bars.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # ============= PRICE FEATURES =============
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # ============= ATR =============
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # ============= MOVING AVERAGES =============
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # ============= VOLATILITY =============
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # ============= RSI =============
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # ============= MACD =============
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ============= BOLLINGER BANDS =============
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + (2 * std20)
        df['bb_lower'] = sma20 - (2 * std20)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ============= VOLUME FEATURES =============
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Extract latest bar features
        latest_idx = len(df) - 1
        latest_features = {}
        
        # Get feature columns (exclude raw OHLC)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in df.columns if c not in exclude_cols 
                       and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        for col in feature_cols:
            value = df[col].iloc[latest_idx]
            latest_features[col] = float(value) if not pd.isna(value) else 0.0
        
        # Add raw OHLC for reference
        latest_features['close'] = float(df['close'].iloc[latest_idx])
        latest_features['open'] = float(df['open'].iloc[latest_idx])
        latest_features['high'] = float(df['high'].iloc[latest_idx])
        latest_features['low'] = float(df['low'].iloc[latest_idx])
        
        return latest_features, df
    
    def validate_features(self, features: Dict[str, float], 
                         expected_feature_cols: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that computed features match model's expected features.
        
        Returns:
            valid: True if all features present
            missing: List of missing feature names
        """
        missing = []
        for col in expected_feature_cols:
            if col not in features:
                missing.append(col)
        
        return len(missing) == 0, missing


class DataFetcher:
    """
    Fetch recent bars from your data source.
    Adapt this to your data infrastructure (database, API, files, etc.)
    """
    
    def __init__(self, feature_store_path: str = None):
        # Auto-detect path
        if feature_store_path:
            self.feature_store = Path(feature_store_path)
        elif Path("feature_store").exists():
            self.feature_store = Path("feature_store")
        else:
            self.feature_store = Path("ML_Trading/feature_store")
    
    def fetch_recent_bars(self, symbol: str, timeframe: str, 
                         n_bars: int = 200) -> pd.DataFrame:
        """
        Fetch the most recent N bars for a symbol/timeframe.
        
        In production, this would:
        - Query your database
        - Call your broker API
        - Read from Redis cache
        - etc.
        
        For now, reads from parquet files.
        """
        # Try nested directory structure first
        file_path = self.feature_store / symbol / f"{symbol}_{timeframe}.parquet"
        
        # Fallback to flat structure
        if not file_path.exists():
            file_path = self.feature_store / f"{symbol}_{timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data not found at:\n  {self.feature_store / symbol / f'{symbol}_{timeframe}.parquet'}\n  or {self.feature_store / f'{symbol}_{timeframe}.parquet'}")
        
        df = pd.read_parquet(file_path)
        
        # Get latest N bars
        df = df.sort_values('timestamp').tail(n_bars).reset_index(drop=True)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


class RealtimeSignalGenerator:
    """
    Complete real-time signal generation system.
    Integrates data fetching, feature computation, and model prediction.
    """
    
    def __init__(self, model_store_path: str = None):
        # Auto-detect paths
        if model_store_path:
            self.model_store = Path(model_store_path)
        elif Path("models").exists():
            self.model_store = Path("models")
        else:
            self.model_store = Path("ML_Trading/models")
        
        self.feature_engine = RealtimeFeatureEngine(lookback_bars=200)
        self.data_fetcher = DataFetcher()
        self.model_cache = {}
    
    def load_model(self, symbol: str, timeframe: str) -> Dict:
        """Load model from disk with caching."""
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # Find latest model file
        pattern = f"{symbol}_{timeframe}_*.pkl"
        model_files = sorted(self.model_store.glob(pattern))
        
        if not model_files:
            raise FileNotFoundError(f"No model found for {symbol} @ {timeframe}")
        
        latest_model = model_files[-1]
        
        import pickle
        with open(latest_model, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_cache[cache_key] = model_data
        return model_data
    
    def generate_signal(self, symbol: str, timeframe: str, 
                       account_equity: float = 100000) -> Dict:
        """
        Generate trading signal with proper feature computation.
        
        Returns complete signal with entry, SL, TP, position size, etc.
        """
        import time
        start_time = time.time()
        
        # Load model
        model_data = self.load_model(symbol, timeframe)
        model = model_data['model']
        threshold = model_data['threshold']
        feature_cols = model_data['feature_cols']
        risk_pct = model_data['risk_pct']
        
        # Fetch recent bars
        recent_bars = self.data_fetcher.fetch_recent_bars(symbol, timeframe, n_bars=200)
        
        # Compute features
        latest_features, df_with_features = self.feature_engine.compute_features(recent_bars)
        
        # Validate features
        valid, missing = self.feature_engine.validate_features(latest_features, feature_cols)
        if not valid:
            raise ValueError(f"Missing features: {missing}")
        
        # Prepare feature vector
        X = np.array([[latest_features.get(col, 0) for col in feature_cols]])
        
        # Generate prediction
        proba = model.predict_proba(X)[0, 1]
        confidence = proba * 100
        
        # Determine signal
        signal = "BUY" if proba >= threshold else "NEUTRAL"
        
        # Position sizing
        entry_price = latest_features['close']
        atr = latest_features.get('atr', entry_price * 0.01)
        
        sl_distance = atr * 1.0  # 1R
        tp_distance = atr * 1.5  # 1.5R
        
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
        
        risk_amount = account_equity * (risk_pct / 100)
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        # Risk/reward
        risk_reward_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': signal,
            'confidence': round(confidence, 2),
            'probability': round(proba, 4),
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'position_size': round(position_size, 4),
            'risk_amount': round(risk_amount, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'atr': round(atr, 5),
            'timestamp': recent_bars['timestamp'].iloc[-1],
            'latency_ms': round(latency_ms, 2),
            'model_metrics': model_data['metrics']
        }


# ==================== EXAMPLE USAGE ====================

def main():
    """Example usage of real-time signal generation."""
    
    generator = RealtimeSignalGenerator()
    
    # Generate signal for XAUUSD @ 15min
    try:
        signal = generator.generate_signal(
            symbol='XAUUSD',
            timeframe='15T',
            account_equity=100000
        )
        
        print("\n" + "="*80)
        print("REAL-TIME SIGNAL")
        print("="*80)
        print(f"\nSymbol: {signal['symbol']}")
        print(f"Timeframe: {signal['timeframe']}")
        print(f"Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']:.1f}%")
        print(f"Probability: {signal['probability']:.4f}")
        print(f"\nEntry Price: {signal['entry_price']:.5f}")
        print(f"Stop Loss: {signal['stop_loss']:.5f}")
        print(f"Take Profit: {signal['take_profit']:.5f}")
        print(f"R:R Ratio: 1:{signal['risk_reward_ratio']:.2f}")
        print(f"\nPosition Size: {signal['position_size']:.4f} units")
        print(f"Risk Amount: ${signal['risk_amount']:.2f}")
        print(f"\nLatency: {signal['latency_ms']:.2f}ms")
        print(f"Timestamp: {signal['timestamp']}")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
