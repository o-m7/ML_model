"""
S5: 2H Momentum ADX+ATR Strategy

Entry Logic:
- Strong trend: ADX ≥ 25 (trending market)
- Volatility: ATR ≥ median (sufficient movement for profit)
- Direction: Price > EMA(50) for longs, < EMA(50) for shorts
- Pullback entry: Wait for price to touch EMA(20) in trending market
- Spread filter: Skip if spread in top 10% of last 60 bars

Exit Logic:
- TP: 1.8×ATR from entry
- SL: EMA(20) cross (trend breaks)
- Time stop: 24 bars (48 hours max hold)

Regime: Trend only (ADX gated)
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy


class S5_MomentumADX(BaseStrategy):
    """2H Momentum: Trade strong trends with ADX+ATR confirmation"""
    
    ALLOWED_REGIMES = ['Trend']  # Only trade in trending markets
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy-specific params
        self.adx_threshold = config.get('adx_threshold', 25)  # Strong trend
        self.atr_percentile = config.get('atr_percentile', 50)  # Median volatility
        self.ema_trend_period = config.get('ema_trend_period', 50)
        self.ema_entry_period = config.get('ema_entry_period', 20)
        self.spread_percentile = config.get('spread_percentile', 90)  # Skip top 10%
        self.lookback_spread = config.get('lookback_spread', 60)
        self.tp_atr_mult = config.get('tp_atr_mult', 1.8)
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum ADX+ATR specific features
        
        Required input features:
        - close, high, low, spread (if available)
        - adx, atr
        - ema_50, ema_20
        """
        df = df.copy()
        
        # Ensure required indicators exist
        if f'ema_{self.ema_trend_period}' not in df.columns:
            df[f'ema_{self.ema_trend_period}'] = df['close'].ewm(span=self.ema_trend_period, adjust=False).mean()
        
        if f'ema_{self.ema_entry_period}' not in df.columns:
            df[f'ema_{self.ema_entry_period}'] = df['close'].ewm(span=self.ema_entry_period, adjust=False).mean()
        
        # ===== TREND STRENGTH FILTER =====
        # Strong trend: ADX ≥ threshold
        df['s5_strong_trend'] = (df['adx'] >= self.adx_threshold).astype(int)
        
        # ADX increasing (trend getting stronger)
        df['s5_adx_rising'] = (df['adx'] > df['adx'].shift(1)).astype(int)
        
        # ===== VOLATILITY FILTER =====
        # ATR percentile threshold
        df['s5_atr_median'] = df['atr'].rolling(window=100).median()
        df['s5_atr_ok'] = (df['atr'] >= df['s5_atr_median']).astype(int)
        
        # ATR not extreme (avoid whipsaws in panic)
        df['s5_atr_90th'] = df['atr'].rolling(window=100).quantile(0.90)
        df['s5_atr_not_extreme'] = (df['atr'] < df['s5_atr_90th']).astype(int)
        
        # ===== DIRECTION FILTER =====
        # Uptrend: Price above EMA(50)
        df['s5_uptrend'] = (df['close'] > df[f'ema_{self.ema_trend_period}']).astype(int)
        
        # Downtrend: Price below EMA(50)
        df['s5_downtrend'] = (df['close'] < df[f'ema_{self.ema_trend_period}']).astype(int)
        
        # Distance from trend EMA (in ATR units)
        df['s5_distance_from_ema50_atr'] = (
            (df['close'] - df[f'ema_{self.ema_trend_period}']) / df['atr']
        )
        
        # ===== ENTRY TIMING (Pullback) =====
        # Distance to entry EMA (pullback target)
        df['s5_distance_to_ema20'] = abs(df['close'] - df[f'ema_{self.ema_entry_period}']) / df['atr']
        
        # Touching EMA(20) = pullback entry
        df['s5_at_ema20'] = (df['s5_distance_to_ema20'] < 0.3).astype(int)
        
        # Bounce from EMA(20): touched last bar, bounced this bar
        df['s5_bounce_from_ema20_up'] = (
            (df['s5_at_ema20'].shift(1) == 1) &
            (df['close'] > df['close'].shift(1)) &
            (df['s5_uptrend'] == 1)
        ).astype(int)
        
        df['s5_bounce_from_ema20_down'] = (
            (df['s5_at_ema20'].shift(1) == 1) &
            (df['close'] < df['close'].shift(1)) &
            (df['s5_downtrend'] == 1)
        ).astype(int)
        
        # ===== SPREAD FILTER =====
        # Calculate spread percentile (if spread available)
        if 'spread' in df.columns:
            df['s5_spread_90th'] = df['spread'].rolling(window=self.lookback_spread).quantile(
                self.spread_percentile / 100.0
            )
            df['s5_spread_ok'] = (df['spread'] < df['s5_spread_90th']).astype(int)
        else:
            # If no spread data, assume OK
            df['s5_spread_ok'] = 1
        
        # ===== ENTRY SIGNALS =====
        # LONG: Strong uptrend + pullback to EMA(20) + bounce + ATR OK + spread OK
        df['s5_long_setup'] = (
            (df['s5_strong_trend'] == 1) &
            (df['s5_uptrend'] == 1) &
            (df['s5_bounce_from_ema20_up'] == 1) &
            (df['s5_atr_ok'] == 1) &
            (df['s5_atr_not_extreme'] == 1) &
            (df['s5_spread_ok'] == 1)
        ).astype(int)
        
        # SHORT: Strong downtrend + pullback to EMA(20) + bounce + ATR OK + spread OK
        df['s5_short_setup'] = (
            (df['s5_strong_trend'] == 1) &
            (df['s5_downtrend'] == 1) &
            (df['s5_bounce_from_ema20_down'] == 1) &
            (df['s5_atr_ok'] == 1) &
            (df['s5_atr_not_extreme'] == 1) &
            (df['s5_spread_ok'] == 1)
        ).astype(int)
        
        # Combined signal
        df['s5_signal_strength'] = df['s5_long_setup'] - df['s5_short_setup']  # +1 long, -1 short, 0 flat
        
        # ===== EXIT SIGNALS =====
        # EMA(20) cross = trend break, exit
        df['s5_ema20_cross_down'] = (
            (df['close'] < df[f'ema_{self.ema_entry_period}']) &
            (df['close'].shift(1) >= df[f'ema_{self.ema_entry_period}'].shift(1))
        ).astype(int)
        
        df['s5_ema20_cross_up'] = (
            (df['close'] > df[f'ema_{self.ema_entry_period}']) &
            (df['close'].shift(1) <= df[f'ema_{self.ema_entry_period}'].shift(1))
        ).astype(int)
        
        # TP distance
        df['s5_tp_distance'] = df['atr'] * self.tp_atr_mult
        
        return df
    
    def get_required_features(self) -> List[str]:
        """Return list of required feature names for this strategy"""
        features = [
            'close', 'high', 'low',
            'adx', 'atr',
            f'ema_{self.ema_trend_period}',
            f'ema_{self.ema_entry_period}',
            's5_strong_trend',
            's5_adx_rising',
            's5_uptrend',
            's5_downtrend',
            's5_at_ema20',
            's5_bounce_from_ema20_up',
            's5_bounce_from_ema20_down',
            's5_long_setup',
            's5_short_setup',
            's5_signal_strength',
            's5_tp_distance',
        ]
        return features
    
    def check_entry_conditions(self, bar: pd.Series, lookback_df: pd.DataFrame) -> Tuple[bool, float]:
        """Check if entry conditions are met (stub for feature engineering)."""
        return False, 0.0
    
    def calculate_exit_levels(self, entry_price: float, atr: float, bar: pd.Series) -> Tuple[float, float]:
        """Calculate TP and SL levels (stub for feature engineering)."""
        tp = entry_price + (atr * self.tp_atr_mult)
        sl_level = bar.get(f'ema_{self.ema_entry_period}', entry_price - atr)
        sl = sl_level if sl_level < entry_price else entry_price - atr
        return tp, sl

