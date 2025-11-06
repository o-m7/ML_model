"""
S6: 4H Multi-Timeframe Trend Alignment Strategy

Entry Logic:
- Daily alignment: Daily EMA(100) trending in same direction as 4H
- 4H signal: Price pulls back to 4H EMA(50) in uptrend
- Momentum: 4H RSI > 55 for longs (> 50 + buffer)
- Volatility check: Daily ATR not in top decile (avoid extreme volatility)
- Entry: On 4H bounce from EMA(50) with daily alignment

Exit Logic:
- TP: 2.5×ATR (wider target for higher TF)
- SL: 4H swing low/high
- Position sizing: Scale down 50% if daily ATR in top decile

Regime: Trend or Neutral (high TF trends are cleaner)
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy


class S6_MultiTFAlignment(BaseStrategy):
    """4H Multi-TF Alignment: Trade 4H with daily confirmation"""
    
    ALLOWED_REGIMES = ['Trend', 'Neutral']
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy-specific params
        self.daily_ema_period = config.get('daily_ema_period', 100)
        self.h4_ema_period = config.get('h4_ema_period', 50)
        self.rsi_long_threshold = config.get('rsi_long_threshold', 55)
        self.rsi_short_threshold = config.get('rsi_short_threshold', 45)
        self.daily_atr_percentile = config.get('daily_atr_percentile', 90)
        self.swing_lookback = config.get('swing_lookback', 50)
        self.tp_atr_mult = config.get('tp_atr_mult', 2.5)
        self.position_scale_down = config.get('position_scale_down', 0.5)  # 50% size if ATR extreme
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-timeframe alignment specific features
        
        Required input features:
        - close, high, low
        - rsi, atr
        - Daily OHLC (for higher TF analysis)
        
        Note: For proper multi-TF, need to resample to daily or join daily bars
        For now, approximate using rolling windows
        """
        df = df.copy()
        
        # ===== DAILY APPROXIMATION (4H bars) =====
        # 6 bars = 24 hours = 1 day
        bars_per_day = 6
        
        # Daily close/high/low (approximate from 4H)
        df['s6_daily_close'] = df['close'].rolling(window=bars_per_day).apply(lambda x: x.iloc[-1], raw=False)
        df['s6_daily_high'] = df['high'].rolling(window=bars_per_day).max()
        df['s6_daily_low'] = df['low'].rolling(window=bars_per_day).min()
        
        # Daily ATR (approximate)
        df['s6_daily_atr'] = df['atr'].rolling(window=bars_per_day).mean() * np.sqrt(bars_per_day)
        
        # Daily EMA(100) - calculated on daily closes
        df['s6_daily_ema'] = df['s6_daily_close'].ewm(span=self.daily_ema_period, adjust=False).mean()
        
        # Daily trend: daily close > daily EMA
        df['s6_daily_uptrend'] = (df['s6_daily_close'] > df['s6_daily_ema']).astype(int)
        df['s6_daily_downtrend'] = (df['s6_daily_close'] < df['s6_daily_ema']).astype(int)
        
        # Daily EMA slope
        df['s6_daily_ema_slope'] = df['s6_daily_ema'].diff(bars_per_day) / df['s6_daily_ema']
        df['s6_daily_trending_up'] = (df['s6_daily_ema_slope'] > 0.001).astype(int)
        df['s6_daily_trending_down'] = (df['s6_daily_ema_slope'] < -0.001).astype(int)
        
        # ===== 4H SETUP =====
        # 4H EMA(50)
        if f'ema_{self.h4_ema_period}' not in df.columns:
            df[f'ema_{self.h4_ema_period}'] = df['close'].ewm(span=self.h4_ema_period, adjust=False).mean()
        
        # 4H trend
        df['s6_h4_uptrend'] = (df['close'] > df[f'ema_{self.h4_ema_period}']).astype(int)
        df['s6_h4_downtrend'] = (df['close'] < df[f'ema_{self.h4_ema_period}']).astype(int)
        
        # Distance to 4H EMA(50)
        df['s6_distance_to_h4_ema'] = abs(df['close'] - df[f'ema_{self.h4_ema_period}']) / df['atr']
        
        # At 4H EMA (pullback)
        df['s6_at_h4_ema'] = (df['s6_distance_to_h4_ema'] < 0.3).astype(int)
        
        # Bounce from 4H EMA
        df['s6_bounce_up'] = (
            (df['s6_at_h4_ema'].shift(1) == 1) &
            (df['close'] > df['close'].shift(1)) &
            (df['s6_h4_uptrend'] == 1)
        ).astype(int)
        
        df['s6_bounce_down'] = (
            (df['s6_at_h4_ema'].shift(1) == 1) &
            (df['close'] < df['close'].shift(1)) &
            (df['s6_h4_downtrend'] == 1)
        ).astype(int)
        
        # ===== ALIGNMENT FILTER =====
        # Both daily and 4H must agree on direction
        df['s6_aligned_up'] = (
            (df['s6_daily_uptrend'] == 1) &
            (df['s6_daily_trending_up'] == 1) &
            (df['s6_h4_uptrend'] == 1)
        ).astype(int)
        
        df['s6_aligned_down'] = (
            (df['s6_daily_downtrend'] == 1) &
            (df['s6_daily_trending_down'] == 1) &
            (df['s6_h4_downtrend'] == 1)
        ).astype(int)
        
        # ===== MOMENTUM FILTER =====
        df['s6_rsi_long_ok'] = (df['rsi'] > self.rsi_long_threshold).astype(int)
        df['s6_rsi_short_ok'] = (df['rsi'] < self.rsi_short_threshold).astype(int)
        
        # ===== VOLATILITY FILTER =====
        # Daily ATR percentile
        df['s6_daily_atr_90th'] = df['s6_daily_atr'].rolling(window=100).quantile(
            self.daily_atr_percentile / 100.0
        )
        df['s6_atr_extreme'] = (df['s6_daily_atr'] >= df['s6_daily_atr_90th']).astype(int)
        df['s6_atr_normal'] = (df['s6_atr_extreme'] == 0).astype(int)
        
        # ===== SWING LEVELS =====
        # For SL placement
        df['s6_swing_high'] = df['high'].rolling(window=self.swing_lookback).max()
        df['s6_swing_low'] = df['low'].rolling(window=self.swing_lookback).min()
        
        # ===== ENTRY SIGNALS =====
        # LONG: Daily + 4H aligned up + pullback + bounce + RSI > 55
        df['s6_long_setup'] = (
            (df['s6_aligned_up'] == 1) &
            (df['s6_bounce_up'] == 1) &
            (df['s6_rsi_long_ok'] == 1)
            # Don't filter by ATR extreme - just scale position size
        ).astype(int)
        
        # SHORT: Daily + 4H aligned down + pullback + bounce + RSI < 45
        df['s6_short_setup'] = (
            (df['s6_aligned_down'] == 1) &
            (df['s6_bounce_down'] == 1) &
            (df['s6_rsi_short_ok'] == 1)
        ).astype(int)
        
        # Combined signal
        df['s6_signal_strength'] = df['s6_long_setup'] - df['s6_short_setup']  # +1 long, -1 short, 0 flat
        
        # ===== POSITION SIZING ADJUSTMENT =====
        # Reduce position size if daily ATR extreme
        df['s6_position_multiplier'] = 1.0
        df.loc[df['s6_atr_extreme'] == 1, 's6_position_multiplier'] = self.position_scale_down
        
        # ===== EXIT TARGETS =====
        # TP distance
        df['s6_tp_distance'] = df['atr'] * self.tp_atr_mult
        
        # SL distance to swing (for risk calculation)
        df['s6_sl_distance_long'] = df['close'] - df['s6_swing_low']
        df['s6_sl_distance_short'] = df['s6_swing_high'] - df['close']
        
        return df
    
    def get_required_features(self) -> List[str]:
        """Return list of required feature names for this strategy"""
        return [
            'close', 'high', 'low',
            'rsi', 'atr',
            f'ema_{self.h4_ema_period}',
            's6_daily_uptrend',
            's6_daily_downtrend',
            's6_daily_trending_up',
            's6_daily_trending_down',
            's6_h4_uptrend',
            's6_h4_downtrend',
            's6_aligned_up',
            's6_aligned_down',
            's6_bounce_up',
            's6_bounce_down',
            's6_long_setup',
            's6_short_setup',
            's6_signal_strength',
            's6_atr_extreme',
            's6_position_multiplier',
            's6_tp_distance',
            's6_sl_distance_long',
            's6_sl_distance_short',
        ]
    
    def check_entry_conditions(self, bar: pd.Series, lookback_df: pd.DataFrame) -> Tuple[bool, float]:
        """Check if entry conditions are met (stub for feature engineering)."""
        return False, 0.0
    
    def calculate_exit_levels(self, entry_price: float, atr: float, bar: pd.Series) -> Tuple[float, float]:
        """Calculate TP and SL levels (stub for feature engineering)."""
        tp = entry_price + (atr * self.tp_atr_mult)
        
        # Use swing levels if available
        swing_low = bar.get('s6_swing_low', entry_price - atr * 1.5)
        swing_high = bar.get('s6_swing_high', entry_price + atr * 1.5)
        
        # For long: SL at swing low, for short: SL at swing high
        sl = swing_low  # Assuming long for this stub
        
        return tp, sl

