"""Regime detection features (volatility, trend strength)."""

import pandas as pd
import numpy as np
from typing import Optional


class RegimeFeatures:
    """Detect market regimes: trending vs ranging, high vs low volatility."""
    
    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all regime features."""
        df = df.copy()
        
        df = self.add_trend_regime(df)
        df = self.add_volatility_regime(df)
        df = self.add_ema_slopes(df)
        
        return df
    
    def add_trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify trend regime using ADX."""
        if 'adx' not in df.columns:
            raise ValueError("ADX required - run FeatureBuilder first")
        
        # Trend strength categories
        df['regime_trend_weak'] = (df['adx'] < 20).astype(int)
        df['regime_trend_moderate'] = ((df['adx'] >= 20) & (df['adx'] < 30)).astype(int)
        df['regime_trend_strong'] = (df['adx'] >= 30).astype(int)
        
        # Trend direction from DI
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            df['regime_trend_up'] = (df['plus_di'] > df['minus_di']).astype(int)
            df['regime_trend_down'] = (df['plus_di'] < df['minus_di']).astype(int)
        
        return df
    
    def add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify volatility regime using ATR percentiles."""
        if 'atr14' not in df.columns:
            raise ValueError("ATR required - run FeatureBuilder first")
        
        # ATR percentile over lookback
        df['atr_percentile'] = df['atr14'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )
        
        # Volatility categories
        df['regime_vol_low'] = (df['atr_percentile'] < 0.33).astype(int)
        df['regime_vol_medium'] = ((df['atr_percentile'] >= 0.33) & 
                                   (df['atr_percentile'] < 0.67)).astype(int)
        df['regime_vol_high'] = (df['atr_percentile'] >= 0.67).astype(int)
        
        # Volatility expansion/contraction
        df['atr_expanding'] = (df['atr14'] > df['atr14'].shift(5)).astype(int)
        
        return df
    
    def add_ema_slopes(self, df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """Add EMA slope features to detect trend momentum."""
        ema_periods = [20, 50, 100, 200]
        
        for period in ema_periods:
            col = f'ema{period}'
            if col in df.columns:
                # Slope (percentage change over lookback)
                df[f'{col}_slope'] = df[col].pct_change(lookback)
                
                # Slope direction
                df[f'{col}_slope_up'] = (df[f'{col}_slope'] > 0).astype(int)
                df[f'{col}_slope_down'] = (df[f'{col}_slope'] < 0).astype(int)
        
        # EMA alignment (all pointing same direction = strong trend)
        if all(f'ema{p}_slope_up' in df.columns for p in [20, 50, 100]):
            df['ema_aligned_up'] = (
                (df['ema20_slope_up'] == 1) & 
                (df['ema50_slope_up'] == 1) & 
                (df['ema100_slope_up'] == 1)
            ).astype(int)
            
            df['ema_aligned_down'] = (
                (df['ema20_slope_down'] == 1) & 
                (df['ema50_slope_down'] == 1) & 
                (df['ema100_slope_down'] == 1)
            ).astype(int)
        
        return df
    
    def add_ranging_detection(self, df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Detect ranging (sideways) markets."""
        # Price staying within narrow range
        high_range = df['high'].rolling(lookback).max()
        low_range = df['low'].rolling(lookback).min()
        range_pct = (high_range - low_range) / df['close']
        
        df['range_pct'] = range_pct
        df['regime_ranging'] = (range_pct < 0.03).astype(int)  # < 3% range
        
        return df

