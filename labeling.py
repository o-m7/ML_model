import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    Implement triple-barrier labeling for trade outcomes.
    Labels based on which barrier is hit first: TP, SL, or time expiry.
    """
    
    def __init__(self, 
                 take_profit_atr: float = 2.0,
                 stop_loss_atr: float = 1.0,
                 time_barrier_bars: int = 24,
                 min_return_threshold: float = 0.0001):
        """
        Args:
            take_profit_atr: ATR multiple for take-profit barrier
            stop_loss_atr: ATR multiple for stop-loss barrier
            time_barrier_bars: Maximum holding period in bars
            min_return_threshold: Minimum return to consider (filter noise)
        """
        self.tp_atr = take_profit_atr
        self.sl_atr = stop_loss_atr
        self.time_barrier = time_barrier_bars
        self.min_return = min_return_threshold
        
    def label_data(self, df: pd.DataFrame, 
                   price_col: str = 'close',
                   atr_col: str = 'ATR') -> pd.DataFrame:
        """
        Apply triple-barrier labeling to dataset.
        
        Returns:
            DataFrame with added columns:
            - label: {1: profitable, 0: timeout, -1: loss}
            - return: Actual return achieved
            - bars_held: Number of bars until barrier hit
            - barrier_hit: {'tp', 'sl', 'time'}
        """
        df = df.copy()
        
        if atr_col not in df.columns:
            raise ValueError(f"ATR column '{atr_col}' not found. Compute ATR first.")
        
        n = len(df)
        labels = np.zeros(n, dtype=np.int8)
        returns = np.zeros(n, dtype=np.float32)
        bars_held = np.zeros(n, dtype=np.int32)
        barrier_hit = np.empty(n, dtype=object)
        
        prices = df[price_col].values
        atr_values = df[atr_col].values
        
        for i in range(n - self.time_barrier):
            entry_price = prices[i]
            atr = atr_values[i]
            
            if pd.isna(entry_price) or pd.isna(atr) or atr == 0:
                continue
            
            # Define barriers
            tp_threshold = entry_price + self.tp_atr * atr
            sl_threshold = entry_price - self.sl_atr * atr
            
            # Scan forward to find which barrier hits first
            hit = False
            for j in range(i + 1, min(i + 1 + self.time_barrier, n)):
                future_price = prices[j]
                
                if pd.isna(future_price):
                    continue
                
                # Check take-profit
                if future_price >= tp_threshold:
                    labels[i] = 1
                    returns[i] = (future_price - entry_price) / entry_price
                    bars_held[i] = j - i
                    barrier_hit[i] = 'tp'
                    hit = True
                    break
                
                # Check stop-loss
                if future_price <= sl_threshold:
                    labels[i] = -1
                    returns[i] = (future_price - entry_price) / entry_price
                    bars_held[i] = j - i
                    barrier_hit[i] = 'sl'
                    hit = True
                    break
            
            # Time barrier (timeout)
            if not hit:
                exit_idx = min(i + self.time_barrier, n - 1)
                exit_price = prices[exit_idx]
                if not pd.isna(exit_price):
                    labels[i] = 0
                    returns[i] = (exit_price - entry_price) / entry_price
                    bars_held[i] = exit_idx - i
                    barrier_hit[i] = 'time'
        
        # Add to dataframe
        df['label'] = labels
        df['return'] = returns
        df['bars_held'] = bars_held
        df['barrier_hit'] = barrier_hit
        
        # Filter low-magnitude moves (noise)
        df.loc[np.abs(df['return']) < self.min_return, 'label'] = 0
        
        # Statistics
        label_counts = df['label'].value_counts()
        logger.info(f"Triple-barrier labeling complete:")
        logger.info(f"  Profitable (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/n*100:.1f}%)")
        logger.info(f"  Loss (-1): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/n*100:.1f}%)")
        logger.info(f"  Timeout (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/n*100:.1f}%)")
        logger.info(f"  Mean return: {df['return'].mean():.4f}")
        logger.info(f"  Mean bars held: {df['bars_held'].mean():.1f}")
        
        return df
    
    def optimize_barriers(self, df: pd.DataFrame, 
                         tp_candidates: list = [1.5, 2.0, 2.5, 3.0],
                         metric: str = 'profit_factor') -> Tuple[float, dict]:
        """
        Grid search over TP/SL ratios to maximize a performance metric.
        
        Returns:
            best_tp_atr: Optimal take-profit ATR multiple
            results: Dict with metrics for each configuration
        """
        results = {}
        
        for tp in tp_candidates:
            self.tp_atr = tp
            labeled_df = self.label_data(df)
            
            # Calculate profit factor
            wins = labeled_df[labeled_df['label'] == 1]['return'].sum()
            losses = abs(labeled_df[labeled_df['label'] == -1]['return'].sum())
            pf = wins / losses if losses > 0 else np.inf
            
            # Win rate
            total_trades = (labeled_df['label'] != 0).sum()
            win_trades = (labeled_df['label'] == 1).sum()
            wr = win_trades / total_trades if total_trades > 0 else 0
            
            # Expected value per trade
            ev = labeled_df[labeled_df['label'] != 0]['return'].mean()
            
            results[tp] = {
                'profit_factor': pf,
                'win_rate': wr,
                'expected_value': ev,
                'total_trades': total_trades
            }
            
            logger.info(f"TP={tp}x ATR: PF={pf:.2f}, WR={wr:.2%}, EV={ev:.4f}, Trades={total_trades}")
        
        # Select best based on metric
        best_tp = max(results.keys(), key=lambda x: results[x][metric])
        logger.info(f"Optimal TP: {best_tp}x ATR (max {metric}={results[best_tp][metric]:.3f})")
        
        return best_tp, results


class MetaLabeler:
    """
    Two-stage labeling:
    1. Primary model predicts direction (long/short)
    2. Meta-model predicts bet size (0 to 1) based on confidence
    """
    
    def __init__(self, primary_labeler: TripleBarrierLabeler):
        self.primary = primary_labeler
        
    def create_meta_labels(self, df: pd.DataFrame, 
                          primary_predictions: np.ndarray) -> pd.DataFrame:
        """
        Given primary model predictions, create meta-labels for sizing.
        
        Meta-label = 1 if primary prediction was correct and profitable
        Meta-label = 0 if primary prediction was wrong or unprofitable
        """
        df = self.primary.label_data(df)
        
        # Meta-label: Did the primary prediction lead to profit?
        meta_labels = np.zeros(len(df))
        
        for i in range(len(df)):
            if primary_predictions[i] == 1 and df.loc[i, 'label'] == 1:
                meta_labels[i] = 1  # Correct long
            elif primary_predictions[i] == -1 and df.loc[i, 'label'] == -1:
                meta_labels[i] = 1  # Correct short (if implemented)
            else:
                meta_labels[i] = 0  # Wrong or no trade
        
        df['meta_label'] = meta_labels
        
        logger.info(f"Meta-labeling: {meta_labels.sum()} correct predictions out of {len(df)}")
        
        return df