"""Common utilities - COMPLETE (NO SMC)."""

import pandas as pd
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for all strategies."""
    
    def __init__(self, config: dict):
        self.config = config
        self.timeframe = config.get('timeframe')
        self.name = self.__class__.__name__
    
    @abstractmethod
    def build_features(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        pass
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        required = self.get_required_features()
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: Missing: {missing}")
        return True
    
    @abstractmethod
    def get_required_features(self) -> list:
        pass


def calculate_breakout_strength(df: pd.DataFrame, atr_col: str = 'atr14') -> pd.Series:
    """✅ SAFE: Current bar O/C known at close."""
    move = df['close'] - df['open']
    strength = move / df[atr_col]
    return strength


def calculate_deviation_from_ma(df: pd.DataFrame, ma_col: str, atr_col: str = 'atr14') -> pd.Series:
    """✅ SAFE"""
    deviation = (df['close'] - df[ma_col]) / df[atr_col]
    return deviation


def detect_consolidation(df: pd.DataFrame, lookback: int = 20, atr_mult: float = 1.5) -> pd.Series:
    """✅ FIXED: Uses shift(1)."""
    high_range = df['high'].shift(1).rolling(lookback).max()
    low_range = df['low'].shift(1).rolling(lookback).min()
    range_size = high_range - low_range
    
    is_consolidation = range_size < (df['atr14'] * atr_mult)
    return is_consolidation.astype(int)


def fibonacci_retracement(high: float, low: float, levels: list = [0.382, 0.5, 0.618]) -> dict:
    """✅ SAFE"""
    diff = high - low
    return {f'fib_{int(level*1000)}': high - (diff * level) for level in levels}


def detect_pullback_to_level(df: pd.DataFrame, level_col: str, tolerance_atr: float = 0.5) -> pd.Series:
    """✅ SAFE"""
    distance = (df['close'] - df[level_col]).abs()
    at_level = distance < (df['atr14'] * tolerance_atr)
    return at_level.astype(int)


def calculate_momentum(df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
    """✅ SAFE"""
    df = df.copy()
    for period in periods:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
    return df


def calculate_volatility_ratio(df: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> pd.Series:
    """✅ SAFE"""
    returns = df['close'].pct_change()
    short_vol = returns.rolling(short_window).std()
    long_vol = returns.rolling(long_window).std()
    vol_ratio = short_vol / long_vol
    return vol_ratio


def safe_rolling_max_min(df: pd.DataFrame, column: str, window: int, 
                         use_high_low: bool = True) -> tuple:
    """✅ SAFE version."""
    if use_high_low and column in ['high', 'low']:
        rolling_max = df[column].shift(1).rolling(window).max()
        rolling_min = df[column].shift(1).rolling(window).min()
    else:
        rolling_max = df[column].rolling(window).max()
        rolling_min = df[column].rolling(window).min()
    
    return rolling_max, rolling_min