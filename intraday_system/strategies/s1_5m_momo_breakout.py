"""S1: 5m Momentum Breakout + Volume Confirmation."""

import pandas as pd
import numpy as np
from typing import Optional
from .common import BaseStrategy, calculate_breakout_strength


class S1_5mMomoBreakout(BaseStrategy):
    """
    5-minute Momentum Breakout with Volume Spike.
    
    Entry conditions:
    - BB width compression (< 20th percentile)
    - Breakout > 1.0 ATR
    - Volume spike > 1.5x MA(20)
    - Momentum alignment (EMA10 > EMA20 for longs)
    """
    
    def build_features(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build S1-specific features."""
        df = df.copy()
        
        # Ensure required base features exist
        required_base = ['bb_width', 'bb_width_pct', 'atr14', 'volume', 'vol_ma20', 'ema10', 'ema20']
        missing = [col for col in required_base if col not in df.columns]
        if missing:
            raise ValueError(f"S1: Missing base features: {missing}. Run FeatureBuilder first.")
        
        # 1. BB Width Compression
        df['s1_bb_compressed'] = (df['bb_width_pct'] < 0.20).astype(int)
        df['s1_bb_width_rank'] = df['bb_width'].rolling(100).rank(pct=True)
        
        # 2. Breakout Strength
        df['s1_breakout_strength'] = calculate_breakout_strength(df)
        df['s1_is_breakout'] = (df['s1_breakout_strength'].abs() > 1.0).astype(int)
        df['s1_breakout_up'] = (df['s1_breakout_strength'] > 1.0).astype(int)
        df['s1_breakout_down'] = (df['s1_breakout_strength'] < -1.0).astype(int)
        
        # 3. Volume Spike
        df['s1_vol_ratio'] = df['volume'] / df['vol_ma20']
        df['s1_vol_spike'] = (df['s1_vol_ratio'] > 1.5).astype(int)
        
        # 4. Momentum Alignment
        df['s1_ema_fast_above_slow'] = (df['ema10'] > df['ema20']).astype(int)
        df['s1_ema_slope'] = df['ema10'].pct_change(5)
        df['s1_ema_momentum_up'] = (df['s1_ema_slope'] > 0).astype(int)
        
        # 5. Combined Signals
        df['s1_long_setup'] = (
            (df['s1_bb_compressed'] == 1) &
            (df['s1_breakout_up'] == 1) &
            (df['s1_vol_spike'] == 1) &
            (df['s1_ema_fast_above_slow'] == 1)
        ).astype(int)
        
        df['s1_short_setup'] = (
            (df['s1_bb_compressed'] == 1) &
            (df['s1_breakout_down'] == 1) &
            (df['s1_vol_spike'] == 1) &
            (df['s1_ema_fast_above_slow'] == 0)
        ).astype(int)
        
        # 6. Additional context features
        df['s1_price_vs_vwap'] = (df['close'] - df['vwap']) / df['atr14'] if 'vwap' in df.columns else 0
        df['s1_rsi14'] = df['rsi14'] if 'rsi14' in df.columns else 50
        
        return df
    
    def get_required_features(self) -> list:
        """Return S1-specific feature names."""
        return [
            's1_bb_compressed',
            's1_bb_width_rank',
            's1_breakout_strength',
            's1_is_breakout',
            's1_breakout_up',
            's1_breakout_down',
            's1_vol_ratio',
            's1_vol_spike',
            's1_ema_fast_above_slow',
            's1_ema_slope',
            's1_ema_momentum_up',
            's1_long_setup',
            's1_short_setup',
            's1_price_vs_vwap',
            's1_rsi14'
        ]

