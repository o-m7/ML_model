"""S5: 2h Momentum with ADX+ATR Regime Filter."""

import pandas as pd
import numpy as np
from typing import Optional
from .common import BaseStrategy


class S5_2hMomoADX(BaseStrategy):
    """
    2-hour Trend Following with Regime Filter.
    
    Entry conditions:
    - ADX > 25 (strong trend)
    - ATR > median (sufficient volatility)
    - EMA50 defines direction
    - EMA20 cross or momentum confirmation
    - Exit on EMA20 cross or 1.5-2x ATR
    """
    
    def build_features(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build S5-specific features."""
        df = df.copy()
        
        # Ensure required base features
        required_base = ['adx', 'atr14', 'ema50', 'ema20', 'plus_di', 'minus_di']
        missing = [col for col in required_base if col not in df.columns]
        if missing:
            raise ValueError(f"S5: Missing base features: {missing}")
        
        # 1. ADX Regime (Strong Trend)
        df['s5_adx_strong'] = (df['adx'] > 25).astype(int)
        df['s5_adx_very_strong'] = (df['adx'] > 35).astype(int)
        
        # 2. ATR Regime (Sufficient Volatility)
        atr_median = df['atr14'].rolling(100).median()
        df['s5_atr_above_median'] = (df['atr14'] > atr_median).astype(int)
        df['s5_atr_percentile'] = df['atr14'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        df['s5_high_volatility'] = (df['s5_atr_percentile'] > 0.5).astype(int)
        
        # 3. Trend Direction (EMA50)
        df['s5_ema50_slope'] = df['ema50'].pct_change(10)
        df['s5_uptrend'] = (df['s5_ema50_slope'] > 0).astype(int)
        df['s5_downtrend'] = (df['s5_ema50_slope'] < 0).astype(int)
        
        # Price position relative to EMA50
        df['s5_above_ema50'] = (df['close'] > df['ema50']).astype(int)
        df['s5_below_ema50'] = (df['close'] < df['ema50']).astype(int)
        
        # 4. Directional Indicators
        df['s5_di_bullish'] = (df['plus_di'] > df['minus_di']).astype(int)
        df['s5_di_bearish'] = (df['plus_di'] < df['minus_di']).astype(int)
        df['s5_di_spread'] = df['plus_di'] - df['minus_di']
        
        # 5. EMA20 Momentum
        df['s5_ema20_slope'] = df['ema20'].pct_change(5)
        df['s5_ema20_up'] = (df['s5_ema20_slope'] > 0).astype(int)
        df['s5_ema20_down'] = (df['s5_ema20_slope'] < 0).astype(int)
        
        # EMA cross
        df['s5_ema20_above_ema50'] = (df['ema20'] > df['ema50']).astype(int)
        df['s5_ema20_cross_up'] = (
            (df['s5_ema20_above_ema50'] == 1) &
            (df['s5_ema20_above_ema50'].shift(1) == 0)
        ).astype(int)
        df['s5_ema20_cross_down'] = (
            (df['s5_ema20_above_ema50'] == 0) &
            (df['s5_ema20_above_ema50'].shift(1) == 1)
        ).astype(int)
        
        # 6. Combined Regime Filter
        df['s5_regime_trending'] = (
            (df['s5_adx_strong'] == 1) &
            (df['s5_high_volatility'] == 1)
        ).astype(int)
        
        # 7. Combined Setups
        # Long: Trending regime + bullish direction + price > EMA50 + momentum
        df['s5_long_setup'] = (
            (df['s5_regime_trending'] == 1) &
            (df['s5_uptrend'] == 1) &
            (df['s5_di_bullish'] == 1) &
            (df['s5_above_ema50'] == 1) &
            ((df['s5_ema20_up'] == 1) | (df['s5_ema20_cross_up'] == 1))
        ).astype(int)
        
        # Short: Trending regime + bearish direction + price < EMA50 + momentum
        df['s5_short_setup'] = (
            (df['s5_regime_trending'] == 1) &
            (df['s5_downtrend'] == 1) &
            (df['s5_di_bearish'] == 1) &
            (df['s5_below_ema50'] == 1) &
            ((df['s5_ema20_down'] == 1) | (df['s5_ema20_cross_down'] == 1))
        ).astype(int)
        
        # 8. Exit signals (for reference, not used in classification but useful)
        df['s5_exit_long'] = (
            (df['s5_ema20_cross_down'] == 1) |
            (df['close'] < df['ema20'])
        ).astype(int)
        
        return df
    
    def get_required_features(self) -> list:
        """Return S5-specific feature names."""
        return [
            's5_adx_strong',
            's5_adx_very_strong',
            's5_atr_above_median',
            's5_atr_percentile',
            's5_high_volatility',
            's5_ema50_slope',
            's5_uptrend',
            's5_downtrend',
            's5_above_ema50',
            's5_below_ema50',
            's5_di_bullish',
            's5_di_bearish',
            's5_di_spread',
            's5_ema20_up',
            's5_ema20_cross_up',
            's5_regime_trending',
            's5_long_setup',
            's5_short_setup'
        ]

