"""Triple-barrier labeling method with ATR-based stops."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class TripleBarrierLabeler:
    """
    Triple-barrier method for generating Up/Down/Flat labels.
    
    Labels based on which barrier (TP or SL) hits first within a horizon.
    - Up (1): TP hits before SL
    - Down (2): SL hits before TP
    - Flat (0): Neither hits or near end of data
    """
    
    def __init__(
        self,
        horizon_bars: int,
        tp_atr_mult: float,
        sl_atr_mult: float,
        atr_col: str = 'atr14'
    ):
        """
        Initialize labeler.
        
        Args:
            horizon_bars: Forward-looking window
            tp_atr_mult: Take profit as ATR multiple
            sl_atr_mult: Stop loss as ATR multiple
            atr_col: ATR column name in DataFrame
        """
        self.horizon_bars = horizon_bars
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.atr_col = atr_col
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create triple-barrier labels.
        
        Args:
            df: DataFrame with OHLCV and ATR
            
        Returns:
            DataFrame with added label columns
        """
        df = df.copy()
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', self.atr_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        labels = []
        returns = []
        durations = []
        tp_hit_flags = []
        sl_hit_flags = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            atr = row[self.atr_col]

            # Handle NaN ATR
            if pd.isna(atr) or atr <= 0:
                atr = row['close'] * 0.02  # Fallback: 2% of price

            # CRITICAL FIX: Entry is at NEXT bar's open (not current bar's close)
            # This matches live trading reality: signal on bar i → enter on bar i+1 at open
            # Check if we have a next bar available
            if i >= len(df) - 1 - self.horizon_bars:
                # Not enough data for next bar entry + full horizon
                labels.append(0)
                returns.append(0.0)
                durations.append(0)
                tp_hit_flags.append(0)
                sl_hit_flags.append(0)
                continue

            # Entry price is NEXT bar's open (realistic!)
            entry_price = df.iloc[i+1]['open']
            tp_price = entry_price + (atr * self.tp_atr_mult)
            sl_price = entry_price - (atr * self.sl_atr_mult)

            # Look ahead from NEXT bar (after entry)
            future = df.iloc[i+1:i+1+self.horizon_bars]
            
            # Find when barriers hit
            tp_hit_idx = future[future['high'] >= tp_price].index
            sl_hit_idx = future[future['low'] <= sl_price].index
            
            tp_hit = len(tp_hit_idx) > 0
            sl_hit = len(sl_hit_idx) > 0
            
            # Determine label
            if tp_hit and sl_hit:
                # Both hit - which came first?
                if tp_hit_idx[0] < sl_hit_idx[0]:
                    label = 1  # Up
                    ret = (tp_price - entry_price) / entry_price
                    dur = tp_hit_idx[0] - i
                    tp_flag, sl_flag = 1, 0
                else:
                    label = 2  # Down
                    ret = (sl_price - entry_price) / entry_price
                    dur = sl_hit_idx[0] - i
                    tp_flag, sl_flag = 0, 1
            elif tp_hit:
                label = 1  # Up
                ret = (tp_price - entry_price) / entry_price
                dur = tp_hit_idx[0] - i
                tp_flag, sl_flag = 1, 0
            elif sl_hit:
                label = 2  # Down
                ret = (sl_price - entry_price) / entry_price
                dur = sl_hit_idx[0] - i
                tp_flag, sl_flag = 0, 1
            else:
                # Neither hit
                label = 0  # Flat
                final_price = future['close'].iloc[-1] if len(future) > 0 else entry_price
                ret = (final_price - entry_price) / entry_price
                dur = len(future)
                tp_flag, sl_flag = 0, 0
            
            labels.append(label)
            returns.append(ret)
            durations.append(dur)
            tp_hit_flags.append(tp_flag)
            sl_hit_flags.append(sl_flag)
        
        # Add to dataframe
        df['target'] = labels
        df['expected_return'] = returns
        df['expected_duration'] = durations
        df['tp_hit'] = tp_hit_flags
        df['sl_hit'] = sl_hit_flags
        
        # Remove last horizon bars (no valid labels)
        df = df.iloc[:-self.horizon_bars].copy()
        
        return df
    
    def get_label_distribution(self, df: pd.DataFrame) -> dict:
        """Get label distribution statistics."""
        if 'target' not in df.columns:
            raise ValueError("No 'target' column found. Run create_labels first.")
        
        total = len(df)
        counts = df['target'].value_counts()
        
        return {
            'total': total,
            'flat_count': int(counts.get(0, 0)),
            'flat_pct': float(counts.get(0, 0) / total * 100),
            'up_count': int(counts.get(1, 0)),
            'up_pct': float(counts.get(1, 0) / total * 100),
            'down_count': int(counts.get(2, 0)),
            'down_pct': float(counts.get(2, 0) / total * 100)
        }

