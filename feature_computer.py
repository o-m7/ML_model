"""
Live Feature Computation Pipeline
==================================

Computes all required features for signal generation in real-time from market data.
Handles both quote features (bid/ask) and OHLCV features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureComputer:
    """Computes all features needed by OHLCV and Quote models."""
    
    def __init__(self):
        """Initialize feature computer."""
        self.quote_data = None
        self.ohlcv_data = {}  # Per-timeframe OHLCV data
        self.last_update = None
        
    def compute_ohlcv_features(self, ohlcv_df: pd.DataFrame, timeframe: str = "1T") -> pd.DataFrame:
        """
        Compute all OHLCV features for a single timeframe.
        
        Args:
            ohlcv_df: DataFrame with columns [timestamp, open, high, low, close, volume]
            timeframe: Timeframe string ("1T", "5T", "15T", "30T")
            
        Returns:
            DataFrame with all OHLCV features (141+ columns)
        """
        df = ohlcv_df.copy()
        
        # Ensure timestamp is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # === BASIC OHLCV FEATURES ===
        df['returns'] = df['close'].pct_change()
        df['high_low_range'] = df['high'] - df['low']
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body_size'] / df['close'] * 100
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['upper_wick_pct'] = df['upper_wick'] / df['high_low_range'] * 100
        df['lower_wick_pct'] = df['lower_wick'] / df['high_low_range'] * 100
        df['true_range'] = np.maximum(
            df['high_low_range'],
            np.maximum(
                abs(df['close'].shift(1) - df['high']),
                abs(df['close'].shift(1) - df['low'])
            )
        )
        
        # === VOLATILITY FEATURES ===
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['parkinson_vol'] = np.log(df['high'] / df['low']).rolling(20).std() * np.sqrt(252 / (4 * np.log(2)))
        df['atr_pct'] = df['true_range'].rolling(14).mean() / df['close'] * 100
        df['volatility'] = df['returns'].rolling(20).std() * 100
        df['vol_ratio'] = df['returns'].rolling(10).std() / df['returns'].rolling(20).std()
        
        # === TREND FEATURES ===
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        df['price_vs_ema_10'] = (df['close'] - df['ema_10']) / df['ema_10'] * 100
        df['price_vs_ema_20'] = (df['close'] - df['ema_20']) / df['ema_20'] * 100
        df['price_vs_ema_50'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100
        df['ema_slope_20'] = (df['ema_20'] - df['ema_20'].shift(5)) / df['ema_20'].shift(5) * 100
        df['ema_cross_10_20'] = (df['ema_10'] > df['ema_20']).astype(int)
        
        # === MOMENTUM FEATURES ===
        df['rsi'] = self._compute_rsi(df['close'], 14)
        df['rsi_ma'] = df['rsi'].rolling(5).mean()
        df['rsi_divergence'] = df['rsi'] - df['rsi'].rolling(20).min()
        
        df['macd'], df['macd_signal'], df['macd_hist'] = self._compute_macd(df['close'])
        df['macd_cross'] = ((df['macd'] > df['macd_signal']).astype(int) - 
                           (df['macd'].shift(1) > df['macd_signal'].shift(1)).astype(int))
        
        for window in [3, 5, 10, 20]:
            df[f'momentum_{window}'] = df['returns'].rolling(window).sum() * 100
            df[f'returns_{window}'] = ((df['close'] / df['close'].shift(window)) - 1) * 100
        
        # === ADX/DI FEATURES ===
        df['adx'], df['plus_di'], df['minus_di'] = self._compute_adx(df)
        df['di_cross'] = (df['plus_di'] > df['minus_di']).astype(int)
        df['trend_strength'] = df['adx']
        
        # === TIME FEATURES ===
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)  # 8am-5pm London
        df['is_us'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)    # 1pm-9pm NY
        
        # === SESSION REGIMES ===
        df['session_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['session_london'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
        df['session_ny'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['session_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 17)).astype(int)
        
        # === VOLATILITY REGIMES ===
        vol_ma = df['volatility'].rolling(50).mean()
        vol_std = df['volatility'].rolling(50).std()
        df['vol_zscore'] = (df['volatility'] - vol_ma) / (vol_std + 1e-9)
        df['regime_high_vol'] = (df['volatility'] > vol_ma + vol_std).astype(int)
        df['regime_low_vol'] = (df['volatility'] < vol_ma - vol_std).astype(int)
        df['regime_volatile'] = (df['vol_zscore'].abs() > 1).astype(int)
        
        # === VOLUME FEATURES ===
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / (df['volume_ma'].std() + 1e-9)
        df['volume_spike'] = (df['volume'] > df['volume_ma'] * 1.5).astype(int)
        df['volume_climax'] = (df['volume'] > df['volume_ma'] * 2).astype(int)
        df['volume_dry'] = (df['volume'] < df['volume_ma'] * 0.5).astype(int)
        df['volume_expansion'] = (df['volume'] > df['volume'].shift(1)).astype(int)
        
        # === PRICE ACTION FEATURES ===
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
        
        # === PULLBACK/CONTINUATION ===
        for window in [5, 10, 20]:
            df[f'pullback_{window}'] = (df['close'] < df['close'].rolling(window).max()).astype(int)
        
        # === VWAP FEATURES ===
        vwap_20 = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_20'] = vwap_20
        df['vwap_deviation_20'] = (df['close'] - vwap_20) / vwap_20 * 100
        df['price_vs_vwap_20'] = (df['close'] > vwap_20).astype(int)
        
        # === FILL COLUMNS ===
        # Fill NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.fillna(0)
        
        return df
    
    def compute_quote_features(self, quote_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all quote features from bid/ask data.
        
        Args:
            quote_df: DataFrame with bid/ask quotes
            
        Returns:
            DataFrame with all quote features (52 columns)
        """
        df = quote_df.copy()
        
        # === BID/ASK AGGREGATION ===
        df['bid_first'] = df.groupby(df.index.date)['bid'].transform('first')
        df['bid_last'] = df.groupby(df.index.date)['bid'].transform('last')
        df['bid_min'] = df.groupby(df.index.date)['bid'].transform('min')
        df['bid_max'] = df.groupby(df.index.date)['bid'].transform('max')
        df['ask_first'] = df.groupby(df.index.date)['ask'].transform('first')
        df['ask_last'] = df.groupby(df.index.date)['ask'].transform('last')
        df['ask_min'] = df.groupby(df.index.date)['ask'].transform('min')
        df['ask_max'] = df.groupby(df.index.date)['ask'].transform('max')
        
        # === MID-PRICE ===
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['mid_price_mean'] = df.groupby(df.index.date)['mid_price'].transform('mean')
        df['mid_price_std'] = df.groupby(df.index.date)['mid_price'].transform('std')
        
        # === LIQUIDITY ===
        df['bid_size_sum'] = df.groupby(df.index.date)['bid_size'].transform('sum')
        df['bid_size_mean'] = df.groupby(df.index.date)['bid_size'].transform('mean')
        df['bid_size_max'] = df.groupby(df.index.date)['bid_size'].transform('max')
        df['ask_size_sum'] = df.groupby(df.index.date)['ask_size'].transform('sum')
        df['ask_size_mean'] = df.groupby(df.index.date)['ask_size'].transform('mean')
        df['ask_size_max'] = df.groupby(df.index.date)['ask_size'].transform('max')
        
        # === SPREAD ===
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = (df['spread'] / df['mid_price']) * 10000  # In basis points
        
        # === PRESSURE ===
        df['buy_pressure'] = df['ask_size'] / (df['bid_size'] + df['ask_size'])
        df['sell_pressure'] = df['bid_size'] / (df['bid_size'] + df['ask_size'])
        
        # === MICROPRICE ===
        df['microprice'] = (df['bid'] * df['ask_size'] + df['ask'] * df['bid_size']) / (df['bid_size'] + df['ask_size'])
        df['microprice_return_1'] = df['microprice'].pct_change().rolling(1).mean() * 100
        df['microprice_return_5'] = df['microprice'].pct_change().rolling(5).mean() * 100
        df['microprice_return_15'] = df['microprice'].pct_change().rolling(15).mean() * 100
        
        # === QUOTE IMBALANCE ===
        df['quote_imbalance'] = (df['ask_size'] - df['bid_size']) / (df['bid_size'] + df['ask_size'])
        
        # === RETURNS ===
        df['bid_return'] = df['bid'].pct_change() * 100
        df['ask_return'] = df['ask'].pct_change() * 100
        df['mid_return'] = df['mid_price'].pct_change() * 100
        
        # === VOLATILITY ===
        df['bid_vol'] = df['bid'].rolling(10).std()
        df['ask_vol'] = df['ask'].rolling(10).std()
        df['mid_vol'] = df['mid_price'].rolling(10).std()
        
        # === SPREAD DYNAMICS ===
        df['spread_zscore'] = (df['spread'] - df['spread'].rolling(20).mean()) / df['spread'].rolling(20).std()
        df['spread_volatility'] = df['spread'].rolling(10).std()
        df['spread_skew'] = df['spread'].rolling(20).skew()
        
        # === ORDERFLOW ===
        df['orderflow_volatility'] = (df['buy_pressure'] - df['sell_pressure']).rolling(10).std()
        df['pressure_gradient'] = (df['buy_pressure'] - df['buy_pressure'].shift(1))
        
        # === ATR FEATURES ===
        df['quote_atr_5'] = (df['ask'] - df['bid']).rolling(5).mean()
        df['quote_atr_10'] = (df['ask'] - df['bid']).rolling(10).mean()
        df['quote_atr_20'] = (df['ask'] - df['bid']).rolling(20).mean()
        df['quote_hl_range'] = df['ask'].rolling(5).max() - df['bid'].rolling(5).min()
        
        # === FILL NaN ===
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.fillna(0)
        
        return df
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> Tuple:
        """Compute ADX and DI indicators."""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = df['true_range'] if 'true_range' in df else (
            df['high'] - df['low']
        )
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        di_ratio = (di_diff / (di_sum + 1e-9)).rolling(period).sum()
        adx = 100 * di_ratio / period
        
        return adx, plus_di, minus_di


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test OHLCV feature computation
    print("Testing OHLCV Feature Computation...")
    print("=" * 80)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    ohlcv_sample = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 3980,
        'high': np.random.randn(100).cumsum() + 3985,
        'low': np.random.randn(100).cumsum() + 3975,
        'close': np.random.randn(100).cumsum() + 3980,
        'volume': np.random.randint(1000, 5000, 100),
        'timestamp': dates
    })
    
    computer = FeatureComputer()
    ohlcv_features = computer.compute_ohlcv_features(ohlcv_sample)
    
    print(f"✓ OHLCV Features computed: {len(ohlcv_features.columns)} columns")
    print(f"  Sample columns: {list(ohlcv_features.columns[:10])}")
    print(f"  Shape: {ohlcv_features.shape}")
    print()
    
    # Test Quote feature computation
    print("Testing Quote Feature Computation...")
    print("=" * 80)
    
    quote_sample = pd.DataFrame({
        'bid': np.random.randn(100).cumsum() + 3980,
        'ask': np.random.randn(100).cumsum() + 3982,
        'bid_size': np.random.randint(100, 500, 100),
        'ask_size': np.random.randint(100, 500, 100),
    }, index=dates)
    
    quote_features = computer.compute_quote_features(quote_sample)
    print(f"✓ Quote Features computed: {len(quote_features.columns)} columns")
    print(f"  Sample columns: {list(quote_features.columns[:10])}")
    print(f"  Shape: {quote_features.shape}")
