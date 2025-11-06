"""
S3: 30m Pullback-to-Trend Strategy

Entry Logic:
- Trend detected: EMA(100) slope > threshold, price above EMA(100)
- Pullback: Price touches EMA(20) OR 38-50% Fibonacci retracement
- RSI filter: RSI > 50 for longs, RSI < 50 for shorts
- Entry: On bounce from support level with volume confirmation

Exit Logic:
- TP: 2.0×ATR OR prior swing high
- SL: 1.2×ATR below pivot/entry

Regime: Trend or Neutral (not Range)
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy


class S3_PullbackTrend(BaseStrategy):
    """30m Pullback-to-Trend: Enter on pullbacks in established trends"""
    
    ALLOWED_REGIMES = ['Trend', 'Neutral']
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy-specific params
        self.ema_trend_period = config.get('ema_trend_period', 100)
        self.ema_pullback_period = config.get('ema_pullback_period', 20)
        self.ema_slope_threshold = config.get('ema_slope_threshold', 0.0001)  # Minimum slope for trend
        self.rsi_long_threshold = config.get('rsi_long_threshold', 50)
        self.rsi_short_threshold = config.get('rsi_short_threshold', 50)
        self.fib_low = config.get('fib_pullback_low', 0.382)  # 38.2% retracement
        self.fib_high = config.get('fib_pullback_high', 0.500)  # 50% retracement
        self.lookback_swing = config.get('lookback_swing', 50)  # Bars to find swing high/low
        self.volume_mult = config.get('volume_mult', 1.0)  # Volume must be > avg
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pullback-to-trend specific features
        
        Required input features:
        - close, high, low, volume
        - rsi, atr
        - ema_{ema_trend_period}, ema_{ema_pullback_period}
        """
        df = df.copy()
        
        # Ensure required base indicators exist
        if f'ema_{self.ema_trend_period}' not in df.columns:
            df[f'ema_{self.ema_trend_period}'] = df['close'].ewm(span=self.ema_trend_period, adjust=False).mean()
        
        if f'ema_{self.ema_pullback_period}' not in df.columns:
            df[f'ema_{self.ema_pullback_period}'] = df['close'].ewm(span=self.ema_pullback_period, adjust=False).mean()
        
        if 'volume_sma_20' not in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # ===== TREND DETECTION =====
        # EMA slope (normalize by price level)
        df['s3_ema_trend_slope'] = (
            df[f'ema_{self.ema_trend_period}'].diff(5) / 
            df[f'ema_{self.ema_trend_period}']
        )
        
        # Price position relative to trend EMA
        df['s3_price_above_trend_ema'] = (df['close'] > df[f'ema_{self.ema_trend_period}']).astype(int)
        df['s3_price_below_trend_ema'] = (df['close'] < df[f'ema_{self.ema_trend_period}']).astype(int)
        
        # Trend strength: distance from EMA in ATR units
        df['s3_distance_from_trend_ema_atr'] = (
            (df['close'] - df[f'ema_{self.ema_trend_period}']) / df['atr']
        )
        
        # Strong uptrend: slope > threshold AND price above EMA
        df['s3_uptrend'] = (
            (df['s3_ema_trend_slope'] > self.ema_slope_threshold) &
            (df['s3_price_above_trend_ema'] == 1)
        ).astype(int)
        
        # Strong downtrend: slope < -threshold AND price below EMA
        df['s3_downtrend'] = (
            (df['s3_ema_trend_slope'] < -self.ema_slope_threshold) &
            (df['s3_price_below_trend_ema'] == 1)
        ).astype(int)
        
        # ===== PULLBACK DETECTION =====
        # Distance to pullback EMA (close to it = pullback)
        df['s3_distance_to_pullback_ema'] = (
            abs(df['close'] - df[f'ema_{self.ema_pullback_period}']) / df['atr']
        )
        df['s3_touching_pullback_ema'] = (df['s3_distance_to_pullback_ema'] < 0.3).astype(int)
        
        # Swing high/low for Fibonacci levels
        df['s3_swing_high'] = df['high'].rolling(window=self.lookback_swing).max()
        df['s3_swing_low'] = df['low'].rolling(window=self.lookback_swing).min()
        df['s3_swing_range'] = df['s3_swing_high'] - df['s3_swing_low']
        
        # Fibonacci retracement levels (38.2% and 50%)
        df['s3_fib_382'] = df['s3_swing_low'] + (df['s3_swing_range'] * self.fib_low)
        df['s3_fib_500'] = df['s3_swing_low'] + (df['s3_swing_range'] * self.fib_high)
        
        # Is price in Fibonacci pullback zone?
        df['s3_in_fib_zone'] = (
            (df['close'] >= df['s3_fib_382']) & 
            (df['close'] <= df['s3_fib_500'])
        ).astype(int)
        
        # Pullback condition: touching EMA OR in Fib zone
        df['s3_is_pullback'] = (
            (df['s3_touching_pullback_ema'] == 1) | 
            (df['s3_in_fib_zone'] == 1)
        ).astype(int)
        
        # ===== BOUNCE DETECTION =====
        # Price bouncing off support: low touched pullback zone, then rallied
        df['s3_bounce_signal'] = 0
        for i in range(2, len(df)):
            # Check if previous bar was at/near support and current bar rallied
            prev_at_support = df.iloc[i-1]['s3_is_pullback'] == 1
            current_rally = df.iloc[i]['close'] > df.iloc[i-1]['close']
            
            if prev_at_support and current_rally:
                df.iloc[i, df.columns.get_loc('s3_bounce_signal')] = 1
        
        # ===== RSI FILTER =====
        df['s3_rsi_long_ok'] = (df['rsi'] > self.rsi_long_threshold).astype(int)
        df['s3_rsi_short_ok'] = (df['rsi'] < self.rsi_short_threshold).astype(int)
        
        # ===== VOLUME CONFIRMATION =====
        df['s3_volume_ok'] = (df['volume'] >= df['volume_sma_20'] * self.volume_mult).astype(int)
        
        # ===== ENTRY SIGNALS =====
        # LONG: Uptrend + pullback + bounce + RSI > 50 + volume
        df['s3_long_setup'] = (
            (df['s3_uptrend'] == 1) &
            (df['s3_is_pullback'] == 1) &
            (df['s3_bounce_signal'] == 1) &
            (df['s3_rsi_long_ok'] == 1) &
            (df['s3_volume_ok'] == 1)
        ).astype(int)
        
        # SHORT: Downtrend + pullback + bounce + RSI < 50 + volume
        df['s3_short_setup'] = (
            (df['s3_downtrend'] == 1) &
            (df['s3_is_pullback'] == 1) &
            (df['s3_bounce_signal'] == 1) &
            (df['s3_rsi_short_ok'] == 1) &
            (df['s3_volume_ok'] == 1)
        ).astype(int)
        
        # Combined signal
        df['s3_signal_strength'] = df['s3_long_setup'] - df['s3_short_setup']  # +1 long, -1 short, 0 flat
        
        return df
    
    def get_required_features(self) -> List[str]:
        """Return list of required feature names for this strategy"""
        return [
            'close', 'high', 'low', 'volume',
            'rsi', 'atr',
            f'ema_{self.ema_trend_period}',
            f'ema_{self.ema_pullback_period}',
            's3_ema_trend_slope',
            's3_uptrend',
            's3_downtrend',
            's3_is_pullback',
            's3_bounce_signal',
            's3_long_setup',
            's3_short_setup',
            's3_signal_strength',
        ]
    
    def check_entry_conditions(self, bar: pd.Series, lookback_df: pd.DataFrame) -> Tuple[bool, float]:
        """Check if entry conditions are met (stub for feature engineering)."""
        return False, 0.0
    
    def calculate_exit_levels(self, entry_price: float, atr: float, bar: pd.Series) -> Tuple[float, float]:
        """Calculate TP and SL levels (stub for feature engineering)."""
        tp = entry_price + (atr * 2.0)
        sl = entry_price - (atr * 1.2)
        return tp, sl

