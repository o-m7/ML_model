"""
S4: 1H Breakout + Retest Strategy

Entry Logic:
- Consolidation: 12-hour range < 1.5×ATR (tight range)
- Breakout: Close beyond range by ≥0.8×ATR (strong move)
- Retest: Price returns to breakout level within 6 bars (validation)
- Confirmation: MACD > 0 OR RSI > 55, Volume ≥ 1.2× average

Exit Logic:
- TP: Range height × 1.5 (measured move)
- SL: Below retest wick
- Cooldown: 3 bars after exit

Regime: Trend or Neutral (breakouts initiate trends)
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy


class S4_BreakoutRetest(BaseStrategy):
    """1H Breakout+Retest: Trade confirmed breakouts after retest"""
    
    ALLOWED_REGIMES = ['Trend', 'Neutral']
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy-specific params
        self.consolidation_period = config.get('consolidation_period', 12)  # 12 bars = 12 hours
        self.consolidation_atr_mult = config.get('consolidation_atr_mult', 1.5)  # Range < 1.5×ATR
        self.breakout_atr_mult = config.get('breakout_atr_mult', 0.8)  # Break by ≥0.8×ATR
        self.retest_bars = config.get('retest_bars', 6)  # Retest within 6 bars
        self.retest_tolerance = config.get('retest_tolerance', 0.3)  # Within 0.3×ATR of breakout level
        self.rsi_threshold = config.get('rsi_threshold', 55)  # RSI > 55 for longs
        self.volume_mult = config.get('volume_mult', 1.2)  # Volume ≥ 1.2× average
        self.tp_mult = config.get('tp_mult', 1.5)  # TP = range × 1.5
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add breakout+retest specific features
        
        Required input features:
        - close, high, low, volume
        - rsi, macd, atr
        """
        df = df.copy()
        
        # Ensure required indicators exist
        if 'volume_sma_20' not in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # ===== CONSOLIDATION DETECTION =====
        # Rolling range over consolidation period
        df['s4_rolling_high'] = df['high'].rolling(window=self.consolidation_period).max()
        df['s4_rolling_low'] = df['low'].rolling(window=self.consolidation_period).min()
        df['s4_rolling_range'] = df['s4_rolling_high'] - df['s4_rolling_low']
        
        # Range threshold: 1.5×ATR
        df['s4_range_threshold'] = df['atr'] * self.consolidation_atr_mult
        
        # Is consolidating? Range < threshold
        df['s4_is_consolidating'] = (
            df['s4_rolling_range'] < df['s4_range_threshold']
        ).astype(int)
        
        # Track consolidation start/end
        df['s4_consolidation_start'] = (
            (df['s4_is_consolidating'] == 1) & 
            (df['s4_is_consolidating'].shift(1) == 0)
        ).astype(int)
        
        # ===== BREAKOUT DETECTION =====
        # Breakout UP: Close above rolling high by ≥ breakout_atr_mult
        df['s4_breakout_up'] = (
            (df['close'] > df['s4_rolling_high'] + (df['atr'] * self.breakout_atr_mult)) &
            (df['s4_is_consolidating'].shift(1) == 1)  # Was consolidating last bar
        ).astype(int)
        
        # Breakout DOWN: Close below rolling low by ≥ breakout_atr_mult
        df['s4_breakout_down'] = (
            (df['close'] < df['s4_rolling_low'] - (df['atr'] * self.breakout_atr_mult)) &
            (df['s4_is_consolidating'].shift(1) == 1)  # Was consolidating last bar
        ).astype(int)
        
        # Store breakout level for retest detection
        df['s4_breakout_level_up'] = np.nan
        df['s4_breakout_level_down'] = np.nan
        df['s4_range_height'] = np.nan
        
        # Track breakout levels
        for i in range(1, len(df)):
            if df.iloc[i]['s4_breakout_up'] == 1:
                df.iloc[i, df.columns.get_loc('s4_breakout_level_up')] = df.iloc[i-1]['s4_rolling_high']
                df.iloc[i, df.columns.get_loc('s4_range_height')] = df.iloc[i-1]['s4_rolling_range']
            elif df.iloc[i]['s4_breakout_down'] == 1:
                df.iloc[i, df.columns.get_loc('s4_breakout_level_down')] = df.iloc[i-1]['s4_rolling_low']
                df.iloc[i, df.columns.get_loc('s4_range_height')] = df.iloc[i-1]['s4_rolling_range']
        
        # Forward fill breakout levels for retest_bars
        df['s4_breakout_level_up'] = df['s4_breakout_level_up'].fillna(method='ffill', limit=self.retest_bars)
        df['s4_breakout_level_down'] = df['s4_breakout_level_down'].fillna(method='ffill', limit=self.retest_bars)
        df['s4_range_height'] = df['s4_range_height'].fillna(method='ffill', limit=self.retest_bars)
        
        # ===== RETEST DETECTION =====
        # Retest UP: After breakout up, price returns near breakout level
        df['s4_retest_up'] = (
            df['s4_breakout_level_up'].notna() &
            (abs(df['low'] - df['s4_breakout_level_up']) <= df['atr'] * self.retest_tolerance)
        ).astype(int)
        
        # Retest DOWN: After breakout down, price returns near breakout level
        df['s4_retest_down'] = (
            df['s4_breakout_level_down'].notna() &
            (abs(df['high'] - df['s4_breakout_level_down']) <= df['atr'] * self.retest_tolerance)
        ).astype(int)
        
        # Retest confirmation: next bar after retest moves in breakout direction
        df['s4_retest_confirmed_up'] = 0
        df['s4_retest_confirmed_down'] = 0
        
        for i in range(1, len(df)):
            # Upside retest: previous bar retested, current bar closes higher
            if df.iloc[i-1]['s4_retest_up'] == 1 and df.iloc[i]['close'] > df.iloc[i-1]['close']:
                df.iloc[i, df.columns.get_loc('s4_retest_confirmed_up')] = 1
            
            # Downside retest: previous bar retested, current bar closes lower
            if df.iloc[i-1]['s4_retest_down'] == 1 and df.iloc[i]['close'] < df.iloc[i-1]['close']:
                df.iloc[i, df.columns.get_loc('s4_retest_confirmed_down')] = 1
        
        # ===== MOMENTUM CONFIRMATION =====
        # MACD positive for longs
        df['s4_macd_long_ok'] = (df['macd'] > 0).astype(int)
        
        # MACD negative for shorts
        df['s4_macd_short_ok'] = (df['macd'] < 0).astype(int)
        
        # RSI confirmation
        df['s4_rsi_long_ok'] = (df['rsi'] > self.rsi_threshold).astype(int)
        df['s4_rsi_short_ok'] = (df['rsi'] < (100 - self.rsi_threshold)).astype(int)
        
        # Either MACD OR RSI must confirm
        df['s4_momentum_long_ok'] = (
            (df['s4_macd_long_ok'] == 1) | (df['s4_rsi_long_ok'] == 1)
        ).astype(int)
        
        df['s4_momentum_short_ok'] = (
            (df['s4_macd_short_ok'] == 1) | (df['s4_rsi_short_ok'] == 1)
        ).astype(int)
        
        # ===== VOLUME CONFIRMATION =====
        df['s4_volume_ok'] = (df['volume'] >= df['volume_sma_20'] * self.volume_mult).astype(int)
        
        # ===== ENTRY SIGNALS =====
        # LONG: Breakout up → retest confirmed → momentum + volume OK
        df['s4_long_setup'] = (
            (df['s4_retest_confirmed_up'] == 1) &
            (df['s4_momentum_long_ok'] == 1) &
            (df['s4_volume_ok'] == 1)
        ).astype(int)
        
        # SHORT: Breakout down → retest confirmed → momentum + volume OK
        df['s4_short_setup'] = (
            (df['s4_retest_confirmed_down'] == 1) &
            (df['s4_momentum_short_ok'] == 1) &
            (df['s4_volume_ok'] == 1)
        ).astype(int)
        
        # Combined signal
        df['s4_signal_strength'] = df['s4_long_setup'] - df['s4_short_setup']  # +1 long, -1 short, 0 flat
        
        # ===== EXIT TARGETS =====
        # TP = range height × 1.5
        df['s4_tp_distance'] = df['s4_range_height'] * self.tp_mult
        
        return df
    
    def get_required_features(self) -> List[str]:
        """Return list of required feature names for this strategy"""
        return [
            'close', 'high', 'low', 'volume',
            'rsi', 'macd', 'atr',
            's4_is_consolidating',
            's4_breakout_up',
            's4_breakout_down',
            's4_retest_confirmed_up',
            's4_retest_confirmed_down',
            's4_long_setup',
            's4_short_setup',
            's4_signal_strength',
            's4_range_height',
            's4_tp_distance',
        ]
    
    def check_entry_conditions(self, bar: pd.Series, lookback_df: pd.DataFrame) -> Tuple[bool, float]:
        """Check if entry conditions are met (stub for feature engineering)."""
        return False, 0.0
    
    def calculate_exit_levels(self, entry_price: float, atr: float, bar: pd.Series) -> Tuple[float, float]:
        """Calculate TP and SL levels (stub for feature engineering)."""
        # Use range height if available, otherwise ATR multiples
        range_height = bar.get('s4_range_height', atr * 1.5)
        tp = entry_price + (range_height * self.tp_mult)
        sl = entry_price - atr  # Below retest wick
        return tp, sl

