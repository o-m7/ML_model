#!/usr/bin/env python3
"""
REGIME CLASSIFIER - Market Regime Detection
============================================

Classifies each bar into Trend/Range/Neutral regimes.
ALL strategies must declare allowed regimes and will be gated.

Regime Definitions:
- Trend: ADX(14) > 25 AND ATR(14) > 50th percentile AND EMA(100) slope in top 40th percentile
- Range: ADX(14) < 20 AND BB width < 40th percentile AND ATR < 50th percentile  
- Neutral: Everything else

Usage:
    from features.regime import RegimeClassifier
    classifier = RegimeClassifier()
    df = classifier.add_regime(df)
    # df now has 'regime' column with values: 'Trend', 'Range', 'Neutral'
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from enum import Enum


class Regime(Enum):
    """Market regime types."""
    TREND = "Trend"
    RANGE = "Range"
    NEUTRAL = "Neutral"


class RegimeClassifier:
    """
    Deterministic regime classifier based on ADX, ATR, MA slope, and Bollinger Bands.
    """
    
    def __init__(self,
                 adx_trend_threshold: float = 25,
                 adx_range_threshold: float = 20,
                 atr_window: int = 14,
                 ema_slope_window: int = 100,
                 bb_window: int = 20,
                 bb_std: float = 2.0):
        """
        Initialize regime classifier.
        
        Args:
            adx_trend_threshold: ADX above this = trending
            adx_range_threshold: ADX below this = ranging
            atr_window: ATR lookback period
            ema_slope_window: EMA period for trend slope
            bb_window: Bollinger Band window
            bb_std: Bollinger Band standard deviations
        """
        self.adx_trend = adx_trend_threshold
        self.adx_range = adx_range_threshold
        self.atr_window = atr_window
        self.ema_window = ema_slope_window
        self.bb_window = bb_window
        self.bb_std = bb_std
    
    def add_regime(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Add regime classification to dataframe.
        
        Args:
            df: DataFrame with OHLCV and indicators
            inplace: Modify df in place or return copy
            
        Returns:
            DataFrame with added 'regime' column
        """
        if not inplace:
            df = df.copy()
        
        # Ensure required indicators exist
        df = self._ensure_indicators(df)
        
        # Calculate regime components
        is_trending = self._detect_trend(df)
        is_ranging = self._detect_range(df)
        
        # Assign regimes (mutually exclusive)
        regime = np.full(len(df), Regime.NEUTRAL.value, dtype=object)
        regime[is_trending] = Regime.TREND.value
        regime[is_ranging & ~is_trending] = Regime.RANGE.value
        
        df['regime'] = regime
        
        # Add confidence scores
        df['regime_confidence'] = self._calculate_confidence(df, is_trending, is_ranging)
        
        return df
    
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are present."""
        
        # ADX
        if 'adx14' not in df.columns and 'adx' not in df.columns:
            df = self._calculate_adx(df, self.atr_window)
        
        # ATR
        if 'atr14' not in df.columns and 'atr' not in df.columns:
            df = self._calculate_atr(df, self.atr_window)
        
        # EMA for trend slope
        if 'ema100' not in df.columns:
            df[f'ema{self.ema_window}'] = df['close'].ewm(span=self.ema_window, adjust=False).mean()
        
        # Bollinger Bands
        if 'bb_width' not in df.columns:
            df = self._calculate_bollinger_width(df, self.bb_window, self.bb_std)
        
        return df
    
    def _detect_trend(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detect trending regime.
        
        Criteria:
        - ADX > 25
        - ATR > 50th percentile
        - EMA slope in top 40th percentile
        """
        adx_col = 'adx14' if 'adx14' in df.columns else 'adx'
        atr_col = 'atr14' if 'atr14' in df.columns else 'atr'
        ema_col = f'ema{self.ema_window}' if f'ema{self.ema_window}' in df.columns else 'ema100'
        
        # ADX > threshold
        adx_high = df[adx_col] > self.adx_trend
        
        # ATR > 50th percentile (rolling)
        atr_median = df[atr_col].rolling(window=100, min_periods=20).median()
        atr_high = df[atr_col] > atr_median
        
        # EMA slope in top 40th percentile
        ema_slope = df[ema_col].diff(5) / df[ema_col].shift(5)  # 5-bar slope
        ema_slope_threshold = ema_slope.rolling(window=100, min_periods=20).quantile(0.60)
        ema_strong = ema_slope.abs() > ema_slope_threshold.abs()
        
        is_trending = adx_high & atr_high & ema_strong
        
        return is_trending.fillna(False).values
    
    def _detect_range(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detect ranging regime.
        
        Criteria:
        - ADX < 20
        - BB width < 40th percentile
        - ATR < 50th percentile
        """
        adx_col = 'adx14' if 'adx14' in df.columns else 'adx'
        atr_col = 'atr14' if 'atr14' in df.columns else 'atr'
        
        # ADX < threshold
        adx_low = df[adx_col] < self.adx_range
        
        # BB width < 40th percentile
        bb_width_threshold = df['bb_width'].rolling(window=100, min_periods=20).quantile(0.40)
        bb_narrow = df['bb_width'] < bb_width_threshold
        
        # ATR < 50th percentile
        atr_median = df[atr_col].rolling(window=100, min_periods=20).median()
        atr_low = df[atr_col] < atr_median
        
        is_ranging = adx_low & bb_narrow & atr_low
        
        return is_ranging.fillna(False).values
    
    def _calculate_confidence(self, df: pd.DataFrame, 
                             is_trending: np.ndarray, 
                             is_ranging: np.ndarray) -> np.ndarray:
        """
        Calculate regime confidence score (0-1).
        
        Higher confidence = more definitive regime signals.
        """
        adx_col = 'adx14' if 'adx14' in df.columns else 'adx'
        atr_col = 'atr14' if 'atr14' in df.columns else 'atr'
        
        confidence = np.zeros(len(df))
        
        # Trend confidence: how far above thresholds
        trend_mask = is_trending
        if trend_mask.any():
            adx_strength = ((df[adx_col] - self.adx_trend) / self.adx_trend).clip(0, 1)
            confidence[trend_mask] = adx_strength[trend_mask].values
        
        # Range confidence: how far below thresholds
        range_mask = is_ranging
        if range_mask.any():
            adx_weakness = ((self.adx_range - df[adx_col]) / self.adx_range).clip(0, 1)
            confidence[range_mask] = adx_weakness[range_mask].values
        
        # Neutral: low confidence
        neutral_mask = ~is_trending & ~is_ranging
        confidence[neutral_mask] = 0.3
        
        return confidence
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        df[f'atr{window}'] = atr
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed indicators
        atr = tr.rolling(window=window).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=window).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=window).mean() / atr
        
        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        df[f'adx{window}'] = adx
        df[f'plus_di{window}'] = plus_di
        df[f'minus_di{window}'] = minus_di
        
        return df
    
    def _calculate_bollinger_width(self, df: pd.DataFrame, 
                                   window: int = 20, 
                                   num_std: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Band width."""
        close = df['close']
        
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        
        # Width as percentage of price
        bb_width = (upper - lower) / sma
        
        df['bb_upper'] = upper
        df['bb_lower'] = lower
        df['bb_mid'] = sma
        df['bb_width'] = bb_width
        
        return df
    
    def get_regime_stats(self, df: pd.DataFrame) -> Dict:
        """Get regime distribution statistics."""
        if 'regime' not in df.columns:
            raise ValueError("DataFrame must have 'regime' column. Call add_regime() first.")
        
        regime_counts = df['regime'].value_counts()
        total = len(df)
        
        stats = {
            'total_bars': total,
            'trend_bars': regime_counts.get(Regime.TREND.value, 0),
            'range_bars': regime_counts.get(Regime.RANGE.value, 0),
            'neutral_bars': regime_counts.get(Regime.NEUTRAL.value, 0),
            'trend_pct': (regime_counts.get(Regime.TREND.value, 0) / total) * 100,
            'range_pct': (regime_counts.get(Regime.RANGE.value, 0) / total) * 100,
            'neutral_pct': (regime_counts.get(Regime.NEUTRAL.value, 0) / total) * 100,
            'avg_confidence': df['regime_confidence'].mean() if 'regime_confidence' in df.columns else None
        }
        
        return stats


def print_regime_report(df: pd.DataFrame, symbol: str = None):
    """Print formatted regime analysis report."""
    classifier = RegimeClassifier()
    stats = classifier.get_regime_stats(df)
    
    print(f"\n{'='*60}")
    print(f"REGIME ANALYSIS{f': {symbol}' if symbol else ''}")
    print(f"{'='*60}")
    print(f"Total Bars:    {stats['total_bars']:,}")
    print(f"")
    print(f"Trend:         {stats['trend_bars']:,} ({stats['trend_pct']:.1f}%)")
    print(f"Range:         {stats['range_bars']:,} ({stats['range_pct']:.1f}%)")
    print(f"Neutral:       {stats['neutral_bars']:,} ({stats['neutral_pct']:.1f}%)")
    print(f"")
    if stats['avg_confidence']:
        print(f"Avg Confidence: {stats['avg_confidence']:.2f}")
    print(f"{'='*60}\n")

