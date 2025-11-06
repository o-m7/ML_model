"""S6: 4h Multi-Timeframe Trend Alignment."""

import pandas as pd
import numpy as np
from typing import Optional
from .common import BaseStrategy


class S6_4hMTF(BaseStrategy):
    """
    4-hour Trend with Daily Filter.
    
    Entry conditions:
    - Daily EMA100 alignment (HTF trend)
    - 4h EMA50 trend confirmation
    - Pullback to 4h EMA20
    - RSI > 55 for longs
    - ATR > 40th percentile
    - Exit at 2-3x ATR or swing points
    """
    
    def build_features(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build S6-specific features."""
        df = df.copy()
        
        # Ensure required base features
        required_base = ['ema100', 'ema50', 'ema20', 'atr14', 'rsi14']
        missing = [col for col in required_base if col not in df.columns]
        if missing:
            raise ValueError(f"S6: Missing base features: {missing}")
        
        # 1. 4H Trend (EMA100 and EMA50)
        df['s6_ema100_slope'] = df['ema100'].pct_change(10)
        df['s6_ema50_slope'] = df['ema50'].pct_change(10)
        
        df['s6_ema100_uptrend'] = (df['s6_ema100_slope'] > 0.001).astype(int)
        df['s6_ema100_downtrend'] = (df['s6_ema100_slope'] < -0.001).astype(int)
        df['s6_ema50_uptrend'] = (df['s6_ema50_slope'] > 0).astype(int)
        df['s6_ema50_downtrend'] = (df['s6_ema50_slope'] < 0).astype(int)
        
        # 2. Trend Alignment
        df['s6_trend_aligned_up'] = (
            (df['s6_ema100_uptrend'] == 1) &
            (df['s6_ema50_uptrend'] == 1)
        ).astype(int)
        
        df['s6_trend_aligned_down'] = (
            (df['s6_ema100_downtrend'] == 1) &
            (df['s6_ema50_downtrend'] == 1)
        ).astype(int)
        
        # 3. Price Position
        df['s6_above_ema50'] = (df['close'] > df['ema50']).astype(int)
        df['s6_below_ema50'] = (df['close'] < df['ema50']).astype(int)
        
        # 4. Pullback to EMA20
        df['s6_dist_to_ema20'] = (df['close'] - df['ema20']) / df['atr14']
        df['s6_at_ema20'] = (df['s6_dist_to_ema20'].abs() < 0.5).astype(int)
        df['s6_near_ema20'] = (df['s6_dist_to_ema20'].abs() < 1.0).astype(int)
        
        # 5. RSI Filter
        df['s6_rsi_strong'] = (df['rsi14'] > 55).astype(int)
        df['s6_rsi_weak'] = (df['rsi14'] < 45).astype(int)
        
        # 6. ATR Filter
        df['s6_atr_percentile'] = df['atr14'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        df['s6_atr_sufficient'] = (df['s6_atr_percentile'] > 0.40).astype(int)
        
        # 7. HTF (Daily) Features if provided
        if htf_df is not None:
            # Assume HTF is daily
            from ..features.utils import align_timeframes
            
            # Select HTF features
            htf_features = ['ema100', 'ema50']
            htf_subset = htf_df[['timestamp'] + [col for col in htf_features if col in htf_df.columns]]
            
            # Align to 4H
            df = align_timeframes(df, htf_subset)
            
            # Rename HTF columns
            if 'ema100_htf' in df.columns:
                df['s6_daily_ema100_slope'] = df['ema100_htf'].pct_change(5)
                df['s6_daily_uptrend'] = (df['s6_daily_ema100_slope'] > 0).astype(int)
                df['s6_daily_downtrend'] = (df['s6_daily_ema100_slope'] < 0).astype(int)
            else:
                # If no HTF data, use 4H as proxy
                df['s6_daily_uptrend'] = df['s6_ema100_uptrend']
                df['s6_daily_downtrend'] = df['s6_ema100_downtrend']
        else:
            # No HTF data - use 4H trend
            df['s6_daily_uptrend'] = df['s6_ema100_uptrend']
            df['s6_daily_downtrend'] = df['s6_ema100_downtrend']
        
        # 8. Full MTF Alignment
        df['s6_mtf_aligned_up'] = (
            (df['s6_daily_uptrend'] == 1) &
            (df['s6_trend_aligned_up'] == 1)
        ).astype(int)
        
        df['s6_mtf_aligned_down'] = (
            (df['s6_daily_downtrend'] == 1) &
            (df['s6_trend_aligned_down'] == 1)
        ).astype(int)
        
        # 9. Combined Setups
        # Long: MTF aligned up + price > EMA50 + pullback to EMA20 + RSI > 55 + ATR sufficient
        df['s6_long_setup'] = (
            (df['s6_mtf_aligned_up'] == 1) &
            (df['s6_above_ema50'] == 1) &
            (df['s6_near_ema20'] == 1) &
            (df['s6_rsi_strong'] == 1) &
            (df['s6_atr_sufficient'] == 1)
        ).astype(int)
        
        # Short: MTF aligned down + price < EMA50 + pullback to EMA20 + RSI < 45 + ATR sufficient
        df['s6_short_setup'] = (
            (df['s6_mtf_aligned_down'] == 1) &
            (df['s6_below_ema50'] == 1) &
            (df['s6_near_ema20'] == 1) &
            (df['s6_rsi_weak'] == 1) &
            (df['s6_atr_sufficient'] == 1)
        ).astype(int)
        
        # 10. Swing points for exits (rolling highs/lows)
        df['s6_swing_high'] = df['high'].rolling(10).max()
        df['s6_swing_low'] = df['low'].rolling(10).min()
        
        return df
    
    def get_required_features(self) -> list:
        """Return S6-specific feature names."""
        return [
            's6_ema100_uptrend',
            's6_ema50_uptrend',
            's6_trend_aligned_up',
            's6_trend_aligned_down',
            's6_above_ema50',
            's6_below_ema50',
            's6_at_ema20',
            's6_near_ema20',
            's6_rsi_strong',
            's6_rsi_weak',
            's6_atr_sufficient',
            's6_daily_uptrend',
            's6_daily_downtrend',
            's6_mtf_aligned_up',
            's6_mtf_aligned_down',
            's6_long_setup',
            's6_short_setup'
        ]

