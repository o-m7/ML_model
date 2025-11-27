"""
Multi-Strategy Backtesting System - Strategy Implementations

Implements 5 different ML-based trading strategies:
1. MLTrendStrategy - Trend following with EMA filter
2. MLMeanReversionStrategy - Mean reversion with Bollinger/VWAP
3. MLBreakoutStrategy - Breakout from volatility squeeze
4. MLRegimeSwitchStrategy - Regime-aware trend/mean reversion
5. MLConfidenceVolFilterStrategy - Confidence-weighted with volatility filter
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StrategyParams:
    """Base parameters for all strategies"""
    timeframe: str
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    max_bars_in_trade: int = 50


class BaseStrategy(ABC):
    """Base class for all ML-based trading strategies"""
    
    def __init__(self, name: str, timeframe: str, params: Dict):
        self.name = name
        self.timeframe = timeframe
        self.params = params
        
        # Set default parameters based on timeframe
        self._set_default_params()
    
    def _set_default_params(self):
        """Set default parameters based on timeframe"""
        tf_num = int(self.timeframe.replace('T', '').replace('H', '00'))
        
        # Default max bars based on timeframe
        if 'max_bars_in_trade' not in self.params:
            if self.timeframe == '5T':
                self.params['max_bars_in_trade'] = 36
            elif self.timeframe == '15T':
                self.params['max_bars_in_trade'] = 24
            elif self.timeframe == '30T':
                self.params['max_bars_in_trade'] = 18
            else:
                self.params['max_bars_in_trade'] = 50
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the dataframe.
        
        Args:
            df: DataFrame with OHLCV, ML predictions (p_up, p_down), and indicators
            
        Returns:
            DataFrame with 'signal' column: +1 (long), -1 (short), 0 (flat)
        """
        pass
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        if 'atr' in df.columns:
            return df['atr']
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _ensure_ml_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure p_up and p_down exist in dataframe"""
        df = df.copy()
        
        # Check if ML predictions exist
        if 'p_up' not in df.columns and 'prediction_prob_long' in df.columns:
            df['p_up'] = df['prediction_prob_long']
            df['p_down'] = 1 - df['prediction_prob_long']
        elif 'p_up' not in df.columns:
            # Default to 0.5 if no predictions
            df['p_up'] = 0.5
            df['p_down'] = 0.5
        
        if 'p_down' not in df.columns:
            df['p_down'] = 1 - df['p_up']
        
        return df


class MLTrendStrategy(BaseStrategy):
    """
    Strategy 1: ML Trend-Following with EMA Filter
    
    Long: p_up >= threshold AND EMA_fast > EMA_slow
    Short: p_down >= threshold AND EMA_fast < EMA_slow
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = {
            'p_long_threshold': 0.60,
            'p_short_threshold': 0.60,
            'tp_atr_mult': 2.0,
            'sl_atr_mult': 1.0,
        }
        
        # Adjust for timeframe
        if timeframe == '5T':
            default_params.update({
                'ema_fast': 20,
                'ema_slow': 50,
                'p_long_threshold': 0.62,
                'p_short_threshold': 0.62,
            })
        elif timeframe == '15T':
            default_params.update({
                'ema_fast': 30,
                'ema_slow': 80,
                'p_long_threshold': 0.60,
                'p_short_threshold': 0.60,
            })
        elif timeframe == '30T':
            default_params.update({
                'ema_fast': 40,
                'ema_slow': 100,
                'p_long_threshold': 0.58,
                'p_short_threshold': 0.58,
            })
        
        if params:
            default_params.update(params)
        
        super().__init__('MLTrendStrategy', timeframe, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-following signals"""
        df = self._ensure_ml_predictions(df)
        df = df.copy()
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=self.params['ema_fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.params['ema_slow'], adjust=False).mean()
        
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        # Long conditions
        long_condition = (
            (df['p_up'] >= self.params['p_long_threshold']) &
            (ema_fast > ema_slow)
        )
        
        # Short conditions
        short_condition = (
            (df['p_down'] >= self.params['p_short_threshold']) &
            (ema_fast < ema_slow)
        )
        
        # Generate signals
        df['signal'] = 0
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # Store entry metadata
        df['tp_atr_mult'] = self.params['tp_atr_mult']
        df['sl_atr_mult'] = self.params['sl_atr_mult']
        df['max_bars'] = self.params['max_bars_in_trade']
        
        return df


class MLMeanReversionStrategy(BaseStrategy):
    """
    Strategy 2: ML-Gated Mean Reversion (VWAP / Bollinger)
    
    Only trade when trend strength is low (ADX < threshold)
    Long: Price closes below lower band AND p_up > 0.5
    Short: Price closes above upper band AND p_down > 0.5
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = {
            'bb_window': 50,
            'bb_std': 2.0,
            'adx_threshold': 20,
            'p_up_threshold': 0.5,
            'tp_atr_mult': 1.5,
            'sl_atr_mult': 1.0,
        }
        
        if params:
            default_params.update(params)
        
        super().__init__('MLMeanReversionStrategy', timeframe, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        df = self._ensure_ml_predictions(df)
        df = df.copy()
        
        # Calculate Bollinger Bands
        sma = df['close'].rolling(window=self.params['bb_window']).mean()
        std = df['close'].rolling(window=self.params['bb_window']).std()
        upper_band = sma + (std * self.params['bb_std'])
        lower_band = sma - (std * self.params['bb_std'])
        
        # Calculate ADX or use existing
        if 'adx' not in df.columns:
            # Simple trend strength proxy
            close_diff = df['close'].diff().abs()
            close_ma = close_diff.rolling(window=14).mean()
            trend_strength = close_ma / df['close'] * 100
            df['adx'] = trend_strength  # Proxy
        else:
            df['adx'] = df['adx'].fillna(0)
        
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        # Only trade in low trend (mean reversion regime)
        low_trend = df['adx'] < self.params['adx_threshold']
        
        # Long: price below lower band AND p_up > threshold
        long_condition = (
            low_trend &
            (df['close'] <= lower_band) &
            (df['p_up'] > self.params['p_up_threshold']) &
            (df['p_up'] >= 0.45)  # Not extremely bearish
        )
        
        # Short: price above upper band AND p_down > threshold
        short_condition = (
            low_trend &
            (df['close'] >= upper_band) &
            (df['p_down'] > self.params['p_up_threshold'])
        )
        
        # Generate signals
        df['signal'] = 0
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # Store TP/SL as mid-line (VWAP) or ATR-based
        df['tp_atr_mult'] = self.params['tp_atr_mult']
        df['sl_atr_mult'] = self.params['sl_atr_mult']
        df['max_bars'] = self.params['max_bars_in_trade']
        df['midline'] = sma  # For mean reversion target
        
        return df


class MLBreakoutStrategy(BaseStrategy):
    """
    Strategy 3: ML-Confirmed Breakout from Volatility Squeeze
    
    Detect squeeze (low volatility), then trade breakouts with ML confirmation
    Long: Break above recent high AND p_up >= threshold
    Short: Break below recent low AND p_down >= threshold
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = {
            'squeeze_percentile': 25,  # Bottom 25% volatility
            'vol_lookback': 100,
            'breakout_lookback': 40,
            'breakout_prob_threshold': 0.65,
            'tp_atr_mult': 3.0,
            'sl_atr_mult': 1.0,
        }
        
        if params:
            default_params.update(params)
        
        super().__init__('MLBreakoutStrategy', timeframe, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout signals"""
        df = self._ensure_ml_predictions(df)
        df = df.copy()
        
        # Calculate Bollinger Band Width as volatility measure
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        bb_width = (std * 2) / sma  # Normalized width
        
        # Detect squeeze (low volatility)
        vol_threshold = bb_width.rolling(window=self.params['vol_lookback']).quantile(
            self.params['squeeze_percentile'] / 100
        )
        is_squeeze = bb_width <= vol_threshold
        
        # Recent high/low
        recent_high = df['high'].rolling(window=self.params['breakout_lookback']).max()
        recent_low = df['low'].rolling(window=self.params['breakout_lookback']).min()
        
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        # Long breakout: close breaks above recent high AND p_up >= threshold
        long_condition = (
            is_squeeze &
            (df['close'] > recent_high.shift(1)) &  # Breakout confirmed
            (df['p_up'] >= self.params['breakout_prob_threshold'])
        )
        
        # Short breakout: close breaks below recent low AND p_down >= threshold
        short_condition = (
            is_squeeze &
            (df['close'] < recent_low.shift(1)) &  # Breakout confirmed
            (df['p_down'] >= self.params['breakout_prob_threshold'])
        )
        
        # Generate signals
        df['signal'] = 0
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # Store breakout levels for SL
        df['tp_atr_mult'] = self.params['tp_atr_mult']
        df['sl_atr_mult'] = self.params['sl_atr_mult']
        df['max_bars'] = self.params['max_bars_in_trade']
        df['breakout_high'] = recent_high
        df['breakout_low'] = recent_low
        
        return df


class MLRegimeSwitchStrategy(BaseStrategy):
    """
    Strategy 4: ML Regime-Switching (Trend vs Mean-Reversion)
    
    Detects regime (trend vs range) and applies appropriate strategy
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        # Initialize sub-strategies
        trend_params = {'p_long_threshold': 0.60, 'p_short_threshold': 0.60}
        mean_revert_params = {'adx_threshold': 20, 'p_up_threshold': 0.5}
        
        default_params = {
            'regime_threshold': 25,  # ADX threshold for regime detection
            'cooldown_bars': 5,
            'trend_params': trend_params,
            'mean_revert_params': mean_revert_params,
        }
        
        if params:
            default_params.update(params)
        
        super().__init__('MLRegimeSwitchStrategy', timeframe, default_params)
        
        # Create sub-strategies
        self.trend_strategy = MLTrendStrategy(timeframe, default_params.get('trend_params', {}))
        self.mean_revert_strategy = MLMeanReversionStrategy(
            timeframe, default_params.get('mean_revert_params', {})
        )
        
        self.last_regime = None
        self.regime_switch_bar = -999
    
    def _detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect regime: 'TREND' or 'RANGE'"""
        if 'adx' not in df.columns:
            # Calculate proxy ADX
            close_diff = df['close'].diff().abs()
            close_ma = close_diff.rolling(window=14).mean()
            trend_strength = close_ma / df['close'] * 100
            adx = trend_strength.fillna(0)
        else:
            adx = df['adx'].fillna(0)
        
        # Regime: TREND if ADX > threshold, else RANGE
        regime = pd.Series('RANGE', index=df.index)
        regime[adx > self.params['regime_threshold']] = 'TREND'
        
        return regime
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on regime"""
        df = self._ensure_ml_predictions(df)
        df = df.copy()
        
        # Detect regime
        df['regime'] = self._detect_regime(df)
        
        # Check for regime switch
        regime_changed = False
        if len(df) > 0:
            current_regime = df['regime'].iloc[-1]
            if self.last_regime and current_regime != self.last_regime:
                regime_changed = True
                self.regime_switch_bar = len(df) - 1
            
            self.last_regime = current_regime
        
        # Apply cooldown after regime switch
        current_bar = len(df) - 1
        in_cooldown = (current_bar - self.regime_switch_bar) < self.params['cooldown_bars']
        
        # Generate signals based on regime
        trend_signals = self.trend_strategy.generate_signals(df)
        mean_revert_signals = self.mean_revert_strategy.generate_signals(df)
        
        # Combine based on regime
        df['signal'] = 0
        trend_mask = (df['regime'] == 'TREND') & ~in_cooldown
        range_mask = (df['regime'] == 'RANGE') & ~in_cooldown
        
        df.loc[trend_mask, 'signal'] = trend_signals.loc[trend_mask, 'signal']
        df.loc[range_mask, 'signal'] = mean_revert_signals.loc[range_mask, 'signal']
        
        # Copy TP/SL from active strategy
        df['tp_atr_mult'] = df['tp_atr_mult'].fillna(2.0)
        df['sl_atr_mult'] = df['sl_atr_mult'].fillna(1.0)
        df['max_bars'] = self.params['max_bars_in_trade']
        
        return df


class MLConfidenceVolFilterStrategy(BaseStrategy):
    """
    Strategy 5: ML Confidence-Weighted Position Sizing + Volatility Filter
    
    Uses ML probability for direction, scales position by confidence
    Only trades in normal volatility conditions
    """
    
    def __init__(self, timeframe: str, params: Optional[Dict] = None):
        default_params = {
            'vol_percentile_low': 20,
            'vol_percentile_high': 80,
            'vol_lookback': 100,
            'base_risk_pct': 0.0025,  # Base risk per trade
            'tp_atr_mult': 2.0,
            'sl_atr_mult': 1.0,
        }
        
        if params:
            default_params.update(params)
        
        super().__init__('MLConfidenceVolFilterStrategy', timeframe, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate confidence-weighted signals"""
        df = self._ensure_ml_predictions(df)
        df = df.copy()
        
        # Calculate volatility (ATR/close)
        atr = self._calculate_atr(df)
        vol_ratio = atr / df['close']
        
        # Volatility filter: only trade in middle volatility range
        vol_low_threshold = vol_ratio.rolling(window=self.params['vol_lookback']).quantile(
            self.params['vol_percentile_low'] / 100
        )
        vol_high_threshold = vol_ratio.rolling(window=self.params['vol_lookback']).quantile(
            self.params['vol_percentile_high'] / 100
        )
        
        vol_normal = (vol_ratio >= vol_low_threshold) & (vol_ratio <= vol_high_threshold)
        
        # Direction based on ML probabilities
        long_condition = (
            vol_normal &
            (df['p_up'] > 0.5)
        )
        
        short_condition = (
            vol_normal &
            (df['p_down'] > 0.5)
        )
        
        # Generate signals
        df['signal'] = 0
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # Position scale based on confidence: 2 * |p_up - 0.5|
        # Higher confidence = bigger position
        df['position_scale'] = 2 * (df['p_up'] - 0.5).abs()
        df['position_scale'] = df['position_scale'].clip(lower=0.5, upper=1.0)  # Cap between 0.5 and 1.0
        
        # Store parameters
        df['tp_atr_mult'] = self.params['tp_atr_mult']
        df['sl_atr_mult'] = self.params['sl_atr_mult']
        df['max_bars'] = self.params['max_bars_in_trade']
        df['base_risk_pct'] = self.params['base_risk_pct']
        
        return df

