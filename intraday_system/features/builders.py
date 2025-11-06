"""Feature builders - COMPLETE (NO SMC)."""

import pandas as pd
import numpy as np
from typing import List


class FeatureBuilder:
    """Build technical indicators."""
    
    def __init__(self):
        self.feature_names = []
    
    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all indicators."""
        df = df.copy()
        
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing: {missing}")
        
        df = self.add_returns(df)
        df = self.add_atr(df, periods=[14, 20])
        df = self.add_ema(df, periods=[10, 20, 50, 100, 200])
        df = self.add_sma(df, periods=[10, 20, 50, 100, 200])
        df = self.add_rsi(df, period=14)
        df = self.add_macd(df)
        df = self.add_bollinger(df)
        df = self.add_donchian(df, period=20)
        df = self.add_vwap(df, window=100)
        df = self.add_adx(df, period=14)
        df = self.add_volatility(df)
        df = self.add_volume_features(df)
        
        return df
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ SAFE"""
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    def add_atr(self, df: pd.DataFrame, periods: List[int] = [14]) -> pd.DataFrame:
        """✅ SAFE"""
        for period in periods:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr{period}'] = tr.rolling(window=period).mean()
            df[f'atr{period}_pct'] = df[f'atr{period}'] / df['close']
        
        return df
    
    def add_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """✅ SAFE"""
        for period in periods:
            df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema{period}_dist'] = (df['close'] - df[f'ema{period}']) / df['close']
        return df
    
    def add_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """✅ SAFE"""
        for period in periods:
            df[f'sma{period}'] = df['close'].rolling(window=period).mean()
            df[f'sma{period}_dist'] = (df['close'] - df[f'sma{period}']) / df['close']
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """✅ SAFE"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi{period}_oversold'] = (df[f'rsi{period}'] < 30).astype(int)
        df[f'rsi{period}_overbought'] = (df[f'rsi{period}'] > 70).astype(int)
        
        return df
    
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """✅ SAFE"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_positive'] = (df['macd'] > 0).astype(int)
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        return df
    
    def add_bollinger(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """✅ SAFE"""
        sma = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = sma + (rolling_std * std)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (rolling_std * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        df['bb_width_pct'] = df['bb_width'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )
        
        return df
    
    def add_donchian(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """✅ FIXED: Uses shift(1)."""
        df['donchian_high'] = df['high'].shift(1).rolling(window=period).max()
        df['donchian_low'] = df['low'].shift(1).rolling(window=period).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
        df['donchian_width'] = (df['donchian_high'] - df['donchian_low']) / df['close']
        
        return df
    
    def add_vwap(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """✅ SAFE"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['close']
        
        return df
    
    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """✅ SAFE"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ SAFE"""
        for window in [10, 20, 50]:
            df[f'vol{window}'] = df['return_1'].rolling(window).std()
        
        df['vol_percentile'] = df['vol20'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )
        
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2).rolling(20).mean()
        )
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ SAFE"""
        for period in [10, 20, 50]:
            df[f'vol_ma{period}'] = df['volume'].rolling(period).mean()
            df[f'vol_ratio{period}'] = df['volume'] / df[f'vol_ma{period}']
        
        df['vol_spike'] = (df['vol_ratio20'] > 1.5).astype(int)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        
        return df