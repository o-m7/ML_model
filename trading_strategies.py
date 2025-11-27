"""
═══════════════════════════════════════════════════════════════════════════════
STRATEGY DEFINITIONS FOR ML-BASED TRADING SYSTEM
═══════════════════════════════════════════════════════════════════════════════

Five concrete strategy implementations using ML predictions:
1. MLTrendStrategy - Trend following with ML + EMA filters
2. MLMeanReversionStrategy - Mean reversion with ML + BB/VWAP
3. MLBreakoutStrategy - Volatility breakout with ML confirmation
4. MLRegimeSwitchStrategy - Adaptive trend/range switching
5. MLConfidenceVolFilterStrategy - Confidence-weighted with vol filter

NOTE: Renamed from strategies.py to trading_strategies.py to avoid conflict
      with existing strategies/ package directory.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════
# BASE STRATEGY CLASS
# ═══════════════════════════════════════════════════════════════════════════

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    All strategies generate signals based on ML predictions and technical indicators.
    Signal conventions:
        +1 = Long
        -1 = Short
         0 = Flat/No position
    """
    
    def __init__(self, name: str, timeframe: str, params: Dict):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            timeframe: Trading timeframe ('5T', '15T', '30T', etc.)
            params: Strategy parameters dictionary
        """
        self.name = name
        self.timeframe = timeframe
        self.params = params
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            df: DataFrame with OHLCV, indicators, and ML predictions
                Must include: open, high, low, close, volume, atr
                Must include: p_up, p_down (ML probabilities)
        
        Returns:
            DataFrame with added 'signal' column and optionally 'position_scale'
        """
        pass
    
    def _validate_dataframe(self, df: pd.DataFrame):
        """Validate that required columns exist."""
        required = ['open', 'high', 'low', 'close', 'volume', 'atr', 'p_up', 'p_down']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: Missing required columns: {missing}")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: ML TREND STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class MLTrendStrategy(BaseStrategy):
    """
    Trend-following strategy using ML predictions with EMA confirmation.
    
    Logic:
        - Long: p_up >= threshold AND fast_EMA > slow_EMA
        - Short: p_down >= threshold AND fast_EMA < slow_EMA
        - Flat: Otherwise
    
    Default parameters by timeframe:
        5T:  fast=10, slow=20, p_long=0.60, p_short=0.60
        15T: fast=20, slow=50, p_long=0.60, p_short=0.60
        30T: fast=20, slow=50, p_long=0.60, p_short=0.60
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = self._get_default_params(timeframe)
        if params:
            default_params.update(params)
        
        super().__init__("MLTrendStrategy", timeframe, default_params)
    
    @staticmethod
    def _get_default_params(timeframe: str) -> Dict:
        """Get default parameters for timeframe."""
        params_by_tf = {
            '5T':  {'ema_fast': 10, 'ema_slow': 20, 'p_long_threshold': 0.60, 
                   'p_short_threshold': 0.60, 'tp_atr_mult': 1.2, 'sl_atr_mult': 1.0, 
                   'max_bars_in_trade': 60},
            '15T': {'ema_fast': 20, 'ema_slow': 50, 'p_long_threshold': 0.60, 
                   'p_short_threshold': 0.60, 'tp_atr_mult': 2.0, 'sl_atr_mult': 1.0, 
                   'max_bars_in_trade': 60},
            '30T': {'ema_fast': 20, 'ema_slow': 50, 'p_long_threshold': 0.60, 
                   'p_short_threshold': 0.60, 'tp_atr_mult': 2.5, 'sl_atr_mult': 1.0, 
                   'max_bars_in_trade': 60}
        }
        return params_by_tf.get(timeframe, params_by_tf['15T'])
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-following signals with ML confirmation."""
        self._validate_dataframe(df)
        
        df = df.copy()
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=self.params['ema_fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.params['ema_slow'], adjust=False).mean()
        
        # Trend direction
        trend_up = ema_fast > ema_slow
        trend_down = ema_fast < ema_slow
        
        # ML confirmation
        ml_long = df['p_up'] >= self.params['p_long_threshold']
        ml_short = df['p_down'] >= self.params['p_short_threshold']
        
        # Generate signals
        df['signal'] = 0
        df.loc[trend_up & ml_long, 'signal'] = 1   # Long
        df.loc[trend_down & ml_short, 'signal'] = -1  # Short
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: ML MEAN REVERSION STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class MLMeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using ML predictions with Bollinger Bands.
    
    Logic:
        - Only trade in low-trend regimes (ADX-like conditions)
        - Long: price < BB lower AND p_up > 0.5
        - Short: price > BB upper AND p_down > 0.5
    
    Default parameters:
        bb_period=20, bb_std=2.0, trend_threshold=0.3
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'trend_threshold': 0.3,  # Max EMA separation for ranging market
            'p_threshold': 0.5,
            'tp_atr_mult': 1.5,
            'sl_atr_mult': 1.0,
            'max_bars_in_trade': 40
        }
        if params:
            default_params.update(params)
        
        super().__init__("MLMeanReversionStrategy", timeframe, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals."""
        self._validate_dataframe(df)
        
        df = df.copy()
        
        # Bollinger Bands
        bb_period = self.params['bb_period']
        bb_std = self.params['bb_std']
        
        sma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        bb_upper = sma + (std * bb_std)
        bb_lower = sma - (std * bb_std)
        
        # Detect ranging regime (low trend strength)
        ema_fast = df['close'].ewm(span=10, adjust=False).mean()
        ema_slow = df['close'].ewm(span=50, adjust=False).mean()
        trend_strength = abs(ema_fast - ema_slow) / ema_slow
        
        ranging = trend_strength < self.params['trend_threshold']
        
        # Price position
        below_lower = df['close'] < bb_lower
        above_upper = df['close'] > bb_upper
        
        # ML confirmation
        ml_long = df['p_up'] > self.params['p_threshold']
        ml_short = df['p_down'] > self.params['p_threshold']
        
        # Generate signals
        df['signal'] = 0
        df.loc[ranging & below_lower & ml_long, 'signal'] = 1   # Long (price oversold)
        df.loc[ranging & above_upper & ml_short, 'signal'] = -1  # Short (price overbought)
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: ML BREAKOUT STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class MLBreakoutStrategy(BaseStrategy):
    """
    Breakout strategy from volatility squeeze with ML confirmation.
    
    Logic:
        - Detect low volatility squeeze
        - Mark recent high/low levels
        - Long: close breaks above resistance AND p_up >= threshold
        - Short: close breaks below support AND p_down >= threshold
    
    Default parameters:
        squeeze_lookback=20, breakout_lookback=20, breakout_prob_threshold=0.65
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = {
            'squeeze_lookback': 20,
            'breakout_lookback': 20,
            'breakout_prob_threshold': 0.65,
            'tp_atr_mult': 2.5,
            'sl_atr_mult': 1.0,
            'max_bars_in_trade': 50
        }
        if params:
            default_params.update(params)
        
        super().__init__("MLBreakoutStrategy", timeframe, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout signals."""
        self._validate_dataframe(df)
        
        df = df.copy()
        
        # Volatility squeeze detection
        squeeze_lookback = self.params['squeeze_lookback']
        atr_ma = df['atr'].rolling(squeeze_lookback).mean()
        vol_percentile = df['atr'].rolling(squeeze_lookback).apply(
            lambda x: (x.iloc[-1] < x).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        low_vol_squeeze = vol_percentile < 0.3  # ATR in bottom 30%
        
        # Recent high/low levels
        breakout_lookback = self.params['breakout_lookback']
        resistance = df['high'].rolling(breakout_lookback).max()
        support = df['low'].rolling(breakout_lookback).min()
        
        # Breakout detection
        break_above = df['close'] > resistance.shift(1)
        break_below = df['close'] < support.shift(1)
        
        # ML confirmation
        ml_long = df['p_up'] >= self.params['breakout_prob_threshold']
        ml_short = df['p_down'] >= self.params['breakout_prob_threshold']
        
        # Generate signals
        df['signal'] = 0
        df.loc[low_vol_squeeze & break_above & ml_long, 'signal'] = 1   # Long breakout
        df.loc[low_vol_squeeze & break_below & ml_short, 'signal'] = -1  # Short breakout
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: ML REGIME SWITCH STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class MLRegimeSwitchStrategy(BaseStrategy):
    """
    Adaptive strategy that switches between trend and mean reversion.
    
    Logic:
        - Compute regime indicator (trend vs range)
        - If trending: apply MLTrendStrategy rules
        - If ranging: apply MLMeanReversionStrategy rules
    
    Regime detection based on ADX-like metric (EMA slope and volatility).
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = {
            'regime_lookback': 20,
            'trend_threshold': 0.4,  # Above = trend, below = range
            'tp_atr_mult': 2.0,
            'sl_atr_mult': 1.0,
            'max_bars_in_trade': 50
        }
        if params:
            default_params.update(params)
        
        super().__init__("MLRegimeSwitchStrategy", timeframe, default_params)
        
        # Initialize sub-strategies
        self.trend_strategy = MLTrendStrategy(timeframe)
        self.mr_strategy = MLMeanReversionStrategy(timeframe)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate adaptive signals based on market regime."""
        self._validate_dataframe(df)
        
        df = df.copy()
        
        # Regime detection
        lookback = self.params['regime_lookback']
        ema_fast = df['close'].ewm(span=10, adjust=False).mean()
        ema_slow = df['close'].ewm(span=50, adjust=False).mean()
        
        # Trend strength metric
        trend_strength = abs(ema_fast - ema_slow) / ema_slow
        
        # Classify regime
        is_trending = trend_strength >= self.params['trend_threshold']
        
        # Get signals from both strategies
        df_trend = self.trend_strategy.generate_signals(df.copy())
        df_mr = self.mr_strategy.generate_signals(df.copy())
        
        # Apply regime-based selection
        df['signal'] = 0
        df.loc[is_trending, 'signal'] = df_trend.loc[is_trending, 'signal']
        df.loc[~is_trending, 'signal'] = df_mr.loc[~is_trending, 'signal']
        
        # Store regime for analysis
        df['regime'] = 'TREND'
        df.loc[~is_trending, 'regime'] = 'RANGE'
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: ML CONFIDENCE VOL FILTER STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class MLConfidenceVolFilterStrategy(BaseStrategy):
    """
    Confidence-weighted strategy with volatility filter.
    
    Logic:
        - Only trade when volatility is in acceptable range
        - Direction: p_up > 0.5 → Long, p_down > 0.5 → Short
        - Position scale: Based on confidence level (distance from 0.5)
    
    Position sizing scales with ML confidence.
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = {
            'vol_lookback': 50,
            'vol_low_percentile': 20,
            'vol_high_percentile': 80,
            'min_confidence': 0.55,
            'tp_atr_mult': 2.0,
            'sl_atr_mult': 1.0,
            'max_bars_in_trade': 50
        }
        if params:
            default_params.update(params)
        
        super().__init__("MLConfidenceVolFilterStrategy", timeframe, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate confidence-weighted signals with vol filter."""
        self._validate_dataframe(df)
        
        df = df.copy()
        
        # Volatility filter
        vol_lookback = self.params['vol_lookback']
        atr_percentile = df['atr'].rolling(vol_lookback).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) * 100 if len(x) > 0 else 50
        )
        
        vol_ok = (
            (atr_percentile >= self.params['vol_low_percentile']) &
            (atr_percentile <= self.params['vol_high_percentile'])
        )
        
        # Confidence calculation
        confidence_long = df['p_up']
        confidence_short = df['p_down']
        
        # Position scale: 0 to 1 based on confidence
        # Scale = 2 * (prob - 0.5), so 0.5→0, 0.75→0.5, 1.0→1.0
        df['position_scale'] = 0.0
        df.loc[df['p_up'] > 0.5, 'position_scale'] = 2 * (df['p_up'] - 0.5)
        df.loc[df['p_down'] > 0.5, 'position_scale'] = 2 * (df['p_down'] - 0.5)
        df['position_scale'] = df['position_scale'].clip(0, 1)
        
        # Generate signals with confidence threshold
        df['signal'] = 0
        
        long_condition = (
            vol_ok & 
            (df['p_up'] > 0.5) & 
            (df['p_up'] >= self.params['min_confidence'])
        )
        short_condition = (
            vol_ok & 
            (df['p_down'] > 0.5) & 
            (df['p_down'] >= self.params['min_confidence'])
        )
        
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    'MLTrendStrategy': MLTrendStrategy,
    'MLMeanReversionStrategy': MLMeanReversionStrategy,
    'MLBreakoutStrategy': MLBreakoutStrategy,
    'MLRegimeSwitchStrategy': MLRegimeSwitchStrategy,
    'MLConfidenceVolFilterStrategy': MLConfidenceVolFilterStrategy
}


def get_all_strategies(timeframe: str) -> list:
    """
    Get instances of all strategies for a given timeframe.
    
    Args:
        timeframe: Trading timeframe
    
    Returns:
        List of strategy instances
    """
    return [strategy_class(timeframe) for strategy_class in STRATEGY_REGISTRY.values()]