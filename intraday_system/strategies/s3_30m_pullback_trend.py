"""S3: 30m Pullback-to-Trend (Trend Continuation)."""

import pandas as pd
import numpy as np
from typing import Optional
from .common import BaseStrategy


class S3_30mPullbackTrend(BaseStrategy):
    """
    30-minute Pullback to Trend Line.
    
    Entry conditions:
    - EMA100 defines trend (slope > 0 for uptrend)
    - Price pulls back to EMA20 or Fib levels (38.2-61.8%)
    - RSI > 50 for longs (maintains trend strength)
    - MACD confirmation
    """
    
    def build_features(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build S3-specific features."""
        df = df.copy()
        
        # Ensure required base features
        required_base = ['ema100', 'ema20', 'atr14', 'rsi14', 'macd', 'macd_hist']
        missing = [col for col in required_base if col not in df.columns]
        if missing:
            raise ValueError(f"S3: Missing base features: {missing}")
        
        # 1. Trend Detection (EMA100 slope)
        df['s3_ema100_slope'] = df['ema100'].pct_change(20)
        df['s3_uptrend'] = (df['s3_ema100_slope'] > 0.001).astype(int)
        df['s3_downtrend'] = (df['s3_ema100_slope'] < -0.001).astype(int)
        
        # 2. Pullback Detection (price near EMA20)
        df['s3_dist_to_ema20'] = (df['close'] - df['ema20']) / df['atr14']
        df['s3_at_ema20'] = (df['s3_dist_to_ema20'].abs() < 0.5).astype(int)
        
        # 3. Price position relative to trend
        df['s3_above_ema100'] = (df['close'] > df['ema100']).astype(int)
        df['s3_below_ema100'] = (df['close'] < df['ema100']).astype(int)
        
        # 4. Fibonacci levels (rolling swing high/low)
        swing_high = df['high'].rolling(20).max()
        swing_low = df['low'].rolling(20).min()
        swing_range = swing_high - swing_low
        
        df['s3_fib_382'] = swing_high - (swing_range * 0.382)
        df['s3_fib_500'] = swing_high - (swing_range * 0.500)
        df['s3_fib_618'] = swing_high - (swing_range * 0.618)
        
        # Distance to Fib levels
        df['s3_near_fib_382'] = ((df['close'] - df['s3_fib_382']).abs() < df['atr14'] * 0.5).astype(int)
        df['s3_near_fib_500'] = ((df['close'] - df['s3_fib_500']).abs() < df['atr14'] * 0.5).astype(int)
        df['s3_near_fib_618'] = ((df['close'] - df['s3_fib_618']).abs() < df['atr14'] * 0.5).astype(int)
        df['s3_near_any_fib'] = (
            df['s3_near_fib_382'] | df['s3_near_fib_500'] | df['s3_near_fib_618']
        ).astype(int)
        
        # 5. RSI Filter (trend continuation)
        df['s3_rsi_above_50'] = (df['rsi14'] > 50).astype(int)
        df['s3_rsi_below_50'] = (df['rsi14'] < 50).astype(int)
        
        # 6. MACD Confirmation
        df['s3_macd_positive'] = (df['macd'] > 0).astype(int)
        df['s3_macd_negative'] = (df['macd'] < 0).astype(int)
        df['s3_macd_hist_positive'] = (df['macd_hist'] > 0).astype(int)
        
        # 7. Combined Setups
        # Long: Uptrend + pullback to support + RSI > 50 + MACD positive
        df['s3_long_setup'] = (
            (df['s3_uptrend'] == 1) &
            (df['s3_above_ema100'] == 1) &
            ((df['s3_at_ema20'] == 1) | (df['s3_near_any_fib'] == 1)) &
            (df['s3_rsi_above_50'] == 1) &
            (df['s3_macd_positive'] == 1)
        ).astype(int)
        
        # Short: Downtrend + pullback to resistance + RSI < 50 + MACD negative
        df['s3_short_setup'] = (
            (df['s3_downtrend'] == 1) &
            (df['s3_below_ema100'] == 1) &
            ((df['s3_at_ema20'] == 1) | (df['s3_near_any_fib'] == 1)) &
            (df['s3_rsi_below_50'] == 1) &
            (df['s3_macd_negative'] == 1)
        ).astype(int)
        
        return df
    
    def get_required_features(self) -> list:
        """Return S3-specific feature names."""
        return [
            's3_ema100_slope',
            's3_uptrend',
            's3_downtrend',
            's3_dist_to_ema20',
            's3_at_ema20',
            's3_above_ema100',
            's3_below_ema100',
            's3_near_any_fib',
            's3_rsi_above_50',
            's3_rsi_below_50',
            's3_macd_positive',
            's3_macd_hist_positive',
            's3_long_setup',
            's3_short_setup'
        ]

