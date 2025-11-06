#!/usr/bin/env python3
"""
S1: 5m Momentum Breakout Strategy (Trend/Neutral Regimes)
==========================================================

Entry Requirements (ALL must be true):
- Squeeze: BB width ≤ 35th percentile (20-bar rolling)
- Breakout: |close - open| ≥ 1.0×ATR
- Volume: volume ≥ 1.5× 20-bar average
- Spread: spread ≤ ATR×0.02
- Regime: Trend OR Neutral (not Range)
- Session: London OR NY overlap (7-22 UTC)

Exit:
- TP: 1.8×ATR from entry
- SL: 0.9×ATR from entry
- Time stop: 10 bars
- Cooldown: 5 bars after exit
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .base import (
    BaseStrategy, StrategyConfig,
    calculate_bb_squeeze, calculate_volume_surge
)


class S1_MomentumBreakout(BaseStrategy):
    """5-minute Momentum Breakout strategy."""
    
    def __init__(self):
        config = StrategyConfig(
            allowed_regimes=['Trend', 'Neutral'],
            session_start=7,   # London open
            session_end=22,    # NY close
            max_spread_atr_mult=0.02,
            tp_atr_mult=1.8,
            sl_atr_mult=0.9,
            cooldown_bars=5,
            max_bars_in_trade=10,  # 10 bars at 5m = 50 minutes
        )
        super().__init__(config)
    
    def check_entry_conditions(self, bar: pd.Series, 
                               lookback_df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check momentum breakout conditions.
        
        Returns:
            (should_enter, confidence_score)
        """
        # Get ATR
        atr = bar.get('atr14', bar.get('atr', 0))
        if atr <= 0:
            return False, 0.0
        
        # 1. Check BB squeeze
        bb_squeeze = calculate_bb_squeeze(lookback_df, window=20, percentile=35)
        if not bb_squeeze.iloc[-1]:
            return False, 0.0
        
        # 2. Check breakout size
        breakout_size = abs(bar['close'] - bar['open'])
        if breakout_size < 1.0 * atr:
            return False, 0.0
        
        # 3. Check volume surge
        vol_surge = calculate_volume_surge(lookback_df, window=20, mult=1.5)
        if not vol_surge.iloc[-1]:
            return False, 0.0
        
        # Calculate confidence score
        # Higher confidence for larger breakouts and higher volume
        breakout_strength = breakout_size / atr
        volume_ratio = bar['volume'] / lookback_df['volume'].rolling(20).mean().iloc[-1]
        
        confidence = min(1.0, (breakout_strength - 1.0) * 0.3 + (volume_ratio - 1.5) * 0.2 + 0.5)
        
        return True, confidence
    
    def calculate_exit_levels(self, entry_price: float, atr: float,
                              bar: pd.Series) -> Tuple[float, float]:
        """Calculate TP and SL for long position."""
        tp_price = entry_price + (atr * self.config.tp_atr_mult)
        sl_price = entry_price - (atr * self.config.sl_atr_mult)
        
        return tp_price, sl_price

