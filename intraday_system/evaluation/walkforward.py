"""Walk-forward cross-validation with purging and embargo."""

import numpy as np
import pandas as pd
from typing import List, Tuple


class WalkForwardCV:
    """Walk-forward time-series cross-validation."""
    
    def __init__(
        self,
        n_folds: int = 10,
        embargo_bars: int = 100,
        purge_bars: int = 50
    ):
        """
        Initialize walk-forward CV.
        
        Args:
            n_folds: Number of folds
            embargo_bars: Bars to embargo after validation
            purge_bars: Bars to purge between train/val
        """
        self.n_folds = n_folds
        self.embargo_bars = embargo_bars
        self.purge_bars = purge_bars
    
    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create walk-forward splits.
        
        Args:
            df: DataFrame with temporal data
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(df)
        fold_size = n_samples // (self.n_folds + 1)
        
        splits = []
        
        for fold_idx in range(self.n_folds):
            # Validation window
            val_start = (fold_idx + 1) * fold_size
            val_end = val_start + fold_size
            
            if val_end > n_samples:
                break
            
            # Train: all data before validation (with purge)
            train_end = val_start - self.purge_bars
            train_indices = np.arange(0, max(0, train_end))
            
            # Validation: with embargo after
            val_indices = np.arange(
                val_start,
                min(val_end, n_samples - self.embargo_bars)
            )
            
            if len(train_indices) > 100 and len(val_indices) > 20:
                splits.append((train_indices, val_indices))
        
        return splits
    
    def get_fold_dates(
        self,
        df: pd.DataFrame,
        splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[dict]:
        """Get date ranges for each fold."""
        fold_info = []
        
        for i, (train_idx, val_idx) in enumerate(splits):
            info = {
                'fold': i + 1,
                'train_start': df.iloc[train_idx[0]]['timestamp'],
                'train_end': df.iloc[train_idx[-1]]['timestamp'],
                'val_start': df.iloc[val_idx[0]]['timestamp'],
                'val_end': df.iloc[val_idx[-1]]['timestamp'],
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            }
            fold_info.append(info)
        
        return fold_info

