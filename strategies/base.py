#!/usr/bin/env python3
"""
Base Strategy Class with RenTec-Grade Filters
==============================================

All strategies must:
1. Declare allowed regimes
2. Implement strict entry filters
3. Respect session windows
4. Check spread caps
5. Honor cooldown periods
6. Calculate TP/SL levels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import time


@dataclass
class StrategyConfig:
    """Configuration for strategy execution."""
    
    # Regime
    allowed_regimes: List[str]  # ['Trend'], ['Range'], ['Trend', 'Neutral'], etc.
    
    # Session filters (UTC hours)
    session_start: int = 0      # Start hour (inclusive)
    session_end: int = 24       # End hour (exclusive)
    avoid_first_minutes: int = 0  # Skip first N minutes of session
    
    # Spread caps
    max_spread_atr_mult: float = 0.02  # Max spread as % of ATR
    
    # Risk parameters
    tp_atr_mult: float = 1.8
    sl_atr_mult: float = 0.9
    cooldown_bars: int = 5
    max_bars_in_trade: int = 50
    
    # Macro blackout (for metals)
    blackout_events: List[str] = None  # ['NFP', 'CPI', 'FOMC']
    blackout_window_minutes: int = 30
    
    def __post_init__(self):
        if self.blackout_events is None:
            self.blackout_events = []


class BaseStrategy(ABC):
    """
    Base class for all RenTec-grade strategies.
    
    Enforces:
    - Regime gating
    - Session filtering
    - Spread checking
    - Cooldown tracking
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.last_exit_bar = -999  # Track for cooldown
        self.name = self.__class__.__name__
    
    @abstractmethod
    def check_entry_conditions(self, bar: pd.Series, lookback_df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check if entry conditions are met.
        
        Args:
            bar: Current bar data
            lookback_df: Historical data for calculations
            
        Returns:
            (should_enter, confidence_score)
        """
        pass
    
    @abstractmethod
    def calculate_exit_levels(self, entry_price: float, atr: float, 
                              bar: pd.Series) -> Tuple[float, float]:
        """
        Calculate TP and SL levels.
        
        Args:
            entry_price: Entry price
            atr: ATR value
            bar: Current bar
            
        Returns:
            (tp_price, sl_price)
        """
        pass
    
    def can_trade(self, bar: pd.Series, current_bar_idx: int, 
                  lookback_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check all filters before allowing trade.
        
        Returns:
            (can_trade, reason_if_not)
        """
        # Check cooldown
        if current_bar_idx - self.last_exit_bar < self.config.cooldown_bars:
            return False, "Cooldown active"
        
        # Check regime
        if 'regime' in bar and bar['regime'] not in self.config.allowed_regimes:
            return False, f"Regime {bar['regime']} not allowed"
        
        # Check session
        if 'timestamp' in bar:
            hour = pd.to_datetime(bar['timestamp']).hour
            if not self._is_in_session(hour):
                return False, "Outside trading session"
        
        # Check spread
        if 'atr14' in bar or 'atr' in bar:
            atr = bar.get('atr14', bar.get('atr', 0))
            max_spread = atr * self.config.max_spread_atr_mult
            
            # Estimate spread from bid/ask if available
            if 'bid' in bar and 'ask' in bar:
                actual_spread = bar['ask'] - bar['bid']
                if actual_spread > max_spread:
                    return False, f"Spread {actual_spread:.5f} > {max_spread:.5f}"
        
        # Check macro blackout (if applicable)
        if self.config.blackout_events and 'timestamp' in bar:
            if self._is_in_blackout(bar['timestamp']):
                return False, "Macro event blackout"
        
        return True, "All filters passed"
    
    def _is_in_session(self, hour: int) -> bool:
        """Check if hour is within trading session."""
        start = self.config.session_start
        end = self.config.session_end
        
        if start < end:
            return start <= hour < end
        else:  # Session wraps midnight
            return hour >= start or hour < end
    
    def _is_in_blackout(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp is within blackout window of macro event.
        
        TODO: Integrate with economic calendar
        Currently just checks known times (placeholder).
        """
        # NFP: First Friday of month, 8:30 AM EST (13:30 UTC)
        # CPI: Mid-month, 8:30 AM EST (13:30 UTC)
        # FOMC: Various, 2:00 PM EST (19:00 UTC)
        
        # Placeholder: Just flag high-volatility hours for metals
        hour = timestamp.hour
        
        if 'NFP' in self.config.blackout_events:
            # Avoid 13:00-14:00 UTC on first Friday
            if timestamp.weekday() == 4 and timestamp.day <= 7:
                if 13 <= hour <= 14:
                    return True
        
        if 'CPI' in self.config.blackout_events:
            # Avoid 13:00-14:00 UTC mid-month
            if 12 <= timestamp.day <= 18:
                if 13 <= hour <= 14:
                    return True
        
        return False
    
    def record_exit(self, exit_bar_idx: int):
        """Record exit for cooldown tracking."""
        self.last_exit_bar = exit_bar_idx
    
    def get_description(self) -> str:
        """Get strategy description."""
        return f"{self.name} - Regimes: {self.config.allowed_regimes}, TP: {self.config.tp_atr_mult}R, SL: {self.config.sl_atr_mult}R"


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr


def calculate_bb_squeeze(df: pd.DataFrame, window: int = 20, percentile: float = 35) -> pd.Series:
    """
    Detect Bollinger Band squeeze.
    
    Returns True when BB width is below specified percentile.
    """
    close = df['close']
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    bb_width = (std * 2) / sma
    threshold = bb_width.rolling(window=100).quantile(percentile / 100)
    
    is_squeeze = bb_width <= threshold
    return is_squeeze


def calculate_volume_surge(df: pd.DataFrame, window: int = 20, mult: float = 1.5) -> pd.Series:
    """Detect volume surge above average."""
    vol_ma = df['volume'].rolling(window=window).mean()
    vol_surge = df['volume'] >= (vol_ma * mult)
    return vol_surge


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

