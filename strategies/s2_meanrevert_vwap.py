#!/usr/bin/env python3
"""
S2: 15m Mean-Revert to VWAP Strategy (Range/Neutral Regimes)
=============================================================

Entry Requirements:
- ADX < 25 (not trending)
- |price - VWAP| ≥ 1.5×ATR (extended)
- RSI: ≥70 for shorts, ≤30 for longs
- Avoid first 30 min of session
- Regime: Range OR Neutral

Exit:
- Target: Price crosses VWAP (mean reversion)
- SL: Last swing high/low ± ATR
- Time stop: 12 bars
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .base import BaseStrategy, StrategyConfig, calculate_rsi, calculate_vwap


class S2_MeanRevertVWAP(BaseStrategy):
    """15-minute Mean-Reversion to VWAP strategy."""
    
    def __init__(self):
        config = StrategyConfig(
            allowed_regimes=['Range', 'Neutral'],
            session_start=7,
            session_end=22,
            avoid_first_minutes=30,
            max_spread_atr_mult=0.02,
            tp_atr_mult=1.2,  # Closer target for mean reversion
            sl_atr_mult=1.0,
            cooldown_bars=3,
            max_bars_in_trade=12,  # 12 bars at 15m = 3 hours
        )
        super().__init__(config)
    
    def check_entry_conditions(self, bar: pd.Series,
                               lookback_df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check mean-reversion conditions.
        
        Returns:
            (should_enter, confidence_score)
        """
        # Get ATR
        atr = bar.get('atr14', bar.get('atr', 0))
        if atr <= 0:
            return False, 0.0
        
        # 1. Check ADX (must be low - not trending)
        adx = bar.get('adx14', bar.get('adx', 100))
        if adx >= 25:
            return False, 0.0
        
        # 2. Calculate VWAP
        vwap = calculate_vwap(lookback_df)
        current_vwap = vwap.iloc[-1]
        
        # Check distance from VWAP
        distance = abs(bar['close'] - current_vwap)
        if distance < 1.5 * atr:
            return False, 0.0
        
        # 3. Check RSI
        rsi = calculate_rsi(lookback_df, window=14)
        current_rsi = rsi.iloc[-1]
        
        # For longs: RSI <= 30 and price below VWAP
        # For shorts: RSI >= 70 and price above VWAP
        is_oversold = current_rsi <= 30 and bar['close'] < current_vwap
        is_overbought = current_rsi >= 70 and bar['close'] > current_vwap
        
        if not (is_oversold or is_overbought):
            return False, 0.0
        
        # Calculate confidence
        # Higher confidence for more extreme RSI and larger VWAP deviation
        rsi_extreme = min(30 - current_rsi, current_rsi - 70) / 20 if (is_oversold or is_overbought) else 0
        vwap_deviation = (distance / atr - 1.5) / 1.5
        
        confidence = min(1.0, 0.5 + rsi_extreme * 0.3 + vwap_deviation * 0.2)
        
        return True, confidence
    
    def calculate_exit_levels(self, entry_price: float, atr: float,
                              bar: pd.Series) -> Tuple[float, float]:
        """Calculate TP and SL for mean-reversion."""
        # For mean reversion, TP is closer (VWAP target)
        # Direction depends on whether we entered from above or below
        
        # Simple approach: TP at 1.2 ATR, SL at 1.0 ATR
        if bar['close'] < entry_price:  # Entered from below (long)
            tp_price = entry_price + (atr * self.config.tp_atr_mult)
            sl_price = entry_price - (atr * self.config.sl_atr_mult)
        else:  # Entered from above (short potential, but we only trade longs for now)
            tp_price = entry_price + (atr * self.config.tp_atr_mult)
            sl_price = entry_price - (atr * self.config.sl_atr_mult)
        
        return tp_price, sl_price

