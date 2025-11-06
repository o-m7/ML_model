"""S4: 1h Breakout + Retest."""

import pandas as pd
import numpy as np
from typing import Optional
from .common import BaseStrategy, detect_consolidation


class S4_1hBreakoutRetest(BaseStrategy):
    """
    1-hour Consolidation Breakout with Retest.
    
    Entry conditions:
    - Multi-hour consolidation (Donchian width < 1.5 ATR)
    - Breakout above/below consolidation
    - Retest of breakout level (within 0.5 ATR)
    - RSI > 55 and MACD positive for longs
    """
    
    def build_features(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build S4-specific features."""
        df = df.copy()
        
        # Ensure required base features
        required_base = ['donchian_high', 'donchian_low', 'donchian_width', 'atr14', 'rsi14', 'macd']
        missing = [col for col in required_base if col not in df.columns]
        if missing:
            raise ValueError(f"S4: Missing base features: {missing}")
        
        # 1. Consolidation Detection
        df['s4_consolidation'] = detect_consolidation(df, lookback=20, atr_mult=1.5)
        df['s4_donchian_range_narrow'] = (df['donchian_width'] < 0.015).astype(int)  # < 1.5%
        
        # 2. Breakout Detection
        df['s4_breakout_high'] = (df['close'] > df['donchian_high'].shift(1)).astype(int)
        df['s4_breakout_low'] = (df['close'] < df['donchian_low'].shift(1)).astype(int)
        
        # Breakout strength
        df['s4_breakout_dist_high'] = (df['close'] - df['donchian_high'].shift(1)) / df['atr14']
        df['s4_breakout_dist_low'] = (df['donchian_low'].shift(1) - df['close']) / df['atr14']
        df['s4_strong_breakout_high'] = (df['s4_breakout_dist_high'] > 0.5).astype(int)
        df['s4_strong_breakout_low'] = (df['s4_breakout_dist_low'] > 0.5).astype(int)
        
        # 3. Retest Detection
        # Store recent breakout levels
        df['s4_recent_breakout_high'] = df['donchian_high'].shift(1).rolling(10).max()
        df['s4_recent_breakout_low'] = df['donchian_low'].shift(1).rolling(10).min()
        
        # Distance to breakout levels
        df['s4_dist_to_bo_high'] = (df['close'] - df['s4_recent_breakout_high']).abs() / df['atr14']
        df['s4_dist_to_bo_low'] = (df['close'] - df['s4_recent_breakout_low']).abs() / df['atr14']
        
        df['s4_retesting_high'] = (df['s4_dist_to_bo_high'] < 0.5).astype(int)
        df['s4_retesting_low'] = (df['s4_dist_to_bo_low'] < 0.5).astype(int)
        
        # 4. RSI Filter
        df['s4_rsi_strong'] = (df['rsi14'] > 55).astype(int)
        df['s4_rsi_weak'] = (df['rsi14'] < 45).astype(int)
        
        # 5. MACD Confirmation
        df['s4_macd_positive'] = (df['macd'] > 0).astype(int)
        df['s4_macd_negative'] = (df['macd'] < 0).astype(int)
        
        # 6. Combined Setups
        # Long: Consolidation -> Breakout high -> Retest -> RSI > 55 + MACD+
        df['s4_long_setup'] = (
            (df['s4_consolidation'].shift(5) == 1) &  # Was consolidating
            (df['s4_breakout_high'].rolling(10).max() == 1) &  # Recent breakout high
            (df['s4_retesting_high'] == 1) &  # Retesting level
            (df['s4_rsi_strong'] == 1) &
            (df['s4_macd_positive'] == 1)
        ).astype(int)
        
        # Short: Consolidation -> Breakout low -> Retest -> RSI < 45 + MACD-
        df['s4_short_setup'] = (
            (df['s4_consolidation'].shift(5) == 1) &
            (df['s4_breakout_low'].rolling(10).max() == 1) &
            (df['s4_retesting_low'] == 1) &
            (df['s4_rsi_weak'] == 1) &
            (df['s4_macd_negative'] == 1)
        ).astype(int)
        
        # 7. Additional context
        df['s4_vol_percentile'] = df['vol_percentile'] if 'vol_percentile' in df.columns else 0.5
        
        return df
    
    def get_required_features(self) -> list:
        """Return S4-specific feature names."""
        return [
            's4_consolidation',
            's4_donchian_range_narrow',
            's4_breakout_high',
            's4_breakout_low',
            's4_strong_breakout_high',
            's4_strong_breakout_low',
            's4_retesting_high',
            's4_retesting_low',
            's4_rsi_strong',
            's4_rsi_weak',
            's4_macd_positive',
            's4_macd_negative',
            's4_long_setup',
            's4_short_setup',
            's4_vol_percentile'
        ]

