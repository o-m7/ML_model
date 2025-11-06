"""S2: 15m Mean-Reversion to VWAP/EMA."""

import pandas as pd
import numpy as np
from typing import Optional
from .common import BaseStrategy, calculate_deviation_from_ma


class S2_15mMeanRevert(BaseStrategy):
    """
    15-minute Mean-Reversion from Extremes.
    
    Entry conditions:
    - Price > 1.5 ATR from VWAP or EMA50
    - RSI extreme (< 30 or > 70)
    - ADX < 30 (ranging market)
    - Expected reversion back to mean
    """
    
    def build_features(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build S2-specific features."""
        df = df.copy()
        
        # Ensure required base features
        required_base = ['vwap', 'ema50', 'atr14', 'rsi14', 'adx']
        missing = [col for col in required_base if col not in df.columns]
        if missing:
            raise ValueError(f"S2: Missing base features: {missing}")
        
        # 1. Deviation from VWAP
        df['s2_vwap_deviation'] = calculate_deviation_from_ma(df, 'vwap')
        df['s2_vwap_far_above'] = (df['s2_vwap_deviation'] > 1.5).astype(int)
        df['s2_vwap_far_below'] = (df['s2_vwap_deviation'] < -1.5).astype(int)
        
        # 2. Deviation from EMA50
        df['s2_ema50_deviation'] = calculate_deviation_from_ma(df, 'ema50')
        df['s2_ema50_far_above'] = (df['s2_ema50_deviation'] > 1.5).astype(int)
        df['s2_ema50_far_below'] = (df['s2_ema50_deviation'] < -1.5).astype(int)
        
        # 3. Combined extreme deviation
        df['s2_extreme_high'] = (
            (df['s2_vwap_far_above'] == 1) | (df['s2_ema50_far_above'] == 1)
        ).astype(int)
        df['s2_extreme_low'] = (
            (df['s2_vwap_far_below'] == 1) | (df['s2_ema50_far_below'] == 1)
        ).astype(int)
        
        # 4. RSI Extremes
        df['s2_rsi_oversold'] = (df['rsi14'] < 30).astype(int)
        df['s2_rsi_overbought'] = (df['rsi14'] > 70).astype(int)
        df['s2_rsi_extreme'] = (df['s2_rsi_oversold'] | df['s2_rsi_overbought']).astype(int)
        
        # 5. Regime Filter (ADX < 30 = ranging)
        df['s2_ranging_regime'] = (df['adx'] < 30).astype(int)
        
        # 6. Mean Reversion Setups
        # Long: Price far below + RSI oversold + ranging
        df['s2_long_setup'] = (
            (df['s2_extreme_low'] == 1) &
            (df['s2_rsi_oversold'] == 1) &
            (df['s2_ranging_regime'] == 1)
        ).astype(int)
        
        # Short: Price far above + RSI overbought + ranging
        df['s2_short_setup'] = (
            (df['s2_extreme_high'] == 1) &
            (df['s2_rsi_overbought'] == 1) &
            (df['s2_ranging_regime'] == 1)
        ).astype(int)
        
        # 7. Additional context
        df['s2_bb_pct'] = df['bb_pct'] if 'bb_pct' in df.columns else 0.5
        df['s2_vol_percentile'] = df['vol_percentile'] if 'vol_percentile' in df.columns else 0.5
        
        return df
    
    def get_required_features(self) -> list:
        """Return S2-specific feature names."""
        return [
            's2_vwap_deviation',
            's2_vwap_far_above',
            's2_vwap_far_below',
            's2_ema50_deviation',
            's2_ema50_far_above',
            's2_ema50_far_below',
            's2_extreme_high',
            's2_extreme_low',
            's2_rsi_oversold',
            's2_rsi_overbought',
            's2_rsi_extreme',
            's2_ranging_regime',
            's2_long_setup',
            's2_short_setup',
            's2_bb_pct',
            's2_vol_percentile'
        ]

