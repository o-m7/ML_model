import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and merge parquet feature stores for XAUUSD.
    Ensures temporal alignment and handles missing data.
    """
    
    def __init__(self, config: Dict):
        self.base_path = Path(config['paths']['base'])
        self.symbol = config['data']['symbol']
        self.primary_tf = config['data']['primary_timeframe']
        self.secondary_tfs = config['data'].get('secondary_timeframes', [])
        
    def load_timeframe_data(self, timeframe: str) -> pd.DataFrame:
        """Load and merge quotes with OHLCV+indicators for a timeframe."""
        dfs = []
        
        # Load quotes (bid/ask/spread) - separate location
        quotes_path = self.base_path / f"quotes/{self.symbol}/{self.symbol}_{timeframe}_quotes.parquet"
        if quotes_path.exists():
            quotes = pd.read_parquet(quotes_path)
            
            # CRITICAL FIX: Ensure timestamp is index before joining
            if 'timestamp' in quotes.columns and not isinstance(quotes.index, pd.DatetimeIndex):
                quotes['timestamp'] = pd.to_datetime(quotes['timestamp'])
                quotes = quotes.set_index('timestamp')
                logger.info(f"Set timestamp as index for quotes data")
            
            quotes = quotes.sort_index()
            
            logger.info(f"Loaded quotes: {len(quotes)} rows, {len(quotes.columns)} columns")
            dfs.append(quotes)
        else:
            logger.warning(f"Quotes file not found: {quotes_path}")
        
        # Load combined OHLCV + indicators - main location
        combined_path = self.base_path / f"{self.symbol}/{self.symbol}_{timeframe}.parquet"
        if combined_path.exists():
            combined = pd.read_parquet(combined_path)
            
            # Ensure timestamp is index
            if 'timestamp' in combined.columns and not isinstance(combined.index, pd.DatetimeIndex):
                combined['timestamp'] = pd.to_datetime(combined['timestamp'])
                combined = combined.set_index('timestamp')
                logger.info(f"Set timestamp as index for main data")
            
            combined = combined.sort_index()
            
            logger.info(f"Loaded OHLCV+indicators: {len(combined)} rows, {len(combined.columns)} features")
            dfs.append(combined)
        else:
            raise FileNotFoundError(f"OHLCV+indicators file not found: {combined_path}")
        
        if not dfs:
            raise FileNotFoundError(f"No data files found for {self.symbol} {timeframe}")
        
        # Merge on timestamp index (both should now have timestamp as index)
        if len(dfs) == 1:
            df = dfs[0]
        else:
            # Use inner join to avoid creating mismatched rows
            df = dfs[0]
            for other in dfs[1:]:
                df = df.join(other, how='inner', rsuffix='_dup')
                # Drop duplicate columns
                dup_cols = [c for c in df.columns if c.endswith('_dup')]
                if dup_cols:
                    logger.warning(f"Dropping duplicate columns: {dup_cols}")
                    df = df.drop(columns=dup_cols)
        
        # Final sort
        df = df.sort_index()
        
        # Check for and remove duplicate timestamps (shouldn't happen now, but safety check)
        if df.index.duplicated().any():
            n_duplicates = df.index.duplicated().sum()
            logger.warning(f"Found {n_duplicates} duplicate timestamps in {timeframe} data - removing duplicates")
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            logger.info(f"After removing duplicates: {len(df)} rows")
        
        # Verify index is monotonic
        if not df.index.is_monotonic_increasing:
            logger.warning(f"Index not monotonic - sorting")
            df = df.sort_index()
        
        # Auto-calculate ATR if missing (CRITICAL for labeling)
        if 'ATR' not in df.columns:
            logger.warning("ATR not found in data - calculating automatically")
            df = self._calculate_atr(df)
        
        logger.info(f"Merged {timeframe} data: {len(df)} rows, {len(df.columns)} features")
        
        return df
        
        logger.info(f"Merged {timeframe} data: {len(df)} rows, {len(df.columns)} features")
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range if missing from data.
        ATR = SMA of True Range over 'period' bars.
        """
        df = df.copy()
        
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("Cannot calculate ATR: missing required columns (high, low, close)")
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # True Range = max of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR = Simple Moving Average of TR
        df['ATR'] = tr.rolling(window=period).mean()
        
        logger.info(f"Calculated ATR: mean={df['ATR'].mean():.4f}, std={df['ATR'].std():.4f}")
        
        return df
    
    def align_timeframes(self, primary_df: pd.DataFrame, secondary_df: pd.DataFrame, 
                        secondary_tf: str) -> pd.DataFrame:
        """
        Align higher timeframe (e.g., 4H) features to primary timeframe (e.g., 1H).
        Uses forward-fill to prevent lookahead bias.
        """
        # Check for duplicates in secondary data before aligning
        if secondary_df.index.duplicated().any():
            n_duplicates = secondary_df.index.duplicated().sum()
            logger.warning(f"Found {n_duplicates} duplicate timestamps in {secondary_tf} data - removing before alignment")
            secondary_df = secondary_df[~secondary_df.index.duplicated(keep='first')]
        
        # Check for duplicates in primary data
        if primary_df.index.duplicated().any():
            n_duplicates = primary_df.index.duplicated().sum()
            logger.warning(f"Found {n_duplicates} duplicate timestamps in primary data - this should have been handled already")
            primary_df = primary_df[~primary_df.index.duplicated(keep='first')]
        
        # Prefix secondary columns to avoid conflicts
        secondary_df = secondary_df.add_prefix(f'{secondary_tf}_')
        
        # Reindex to primary timeframe with forward fill (no lookahead)
        aligned = secondary_df.reindex(primary_df.index, method='ffill')
        
        # Merge with primary
        result = primary_df.join(aligned, how='left')
        
        logger.info(f"Aligned {secondary_tf} features: added {len(aligned.columns)} columns")
        return result
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge all timeframes with proper alignment."""
        # Load primary timeframe
        df = self.load_timeframe_data(self.primary_tf)
        
        # Add higher timeframe features for regime detection
        for secondary_tf in self.secondary_tfs:
            secondary_df = self.load_timeframe_data(secondary_tf)
            df = self.align_timeframes(df, secondary_df, secondary_tf)
        
        # Add temporal features (no lookahead)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_london'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['is_newyork'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
        df['is_asian'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        
        logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} features")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def split_train_val_test(self, df: pd.DataFrame, 
                             train_months: int, 
                             val_months: int, 
                             test_months: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create initial train/val/test split based on time.
        Walk-forward validation will create multiple folds from this.
        """
        total_months = train_months + val_months + test_months
        total_days = total_months * 30
        
        if len(df) < total_days * 24:  # Rough check for H1 data
            logger.warning(f"Dataset may be too small for requested split: {len(df)} bars")
        
        # Calculate split points
        train_end_idx = len(df) - (val_months + test_months) * 30 * 24
        val_end_idx = len(df) - test_months * 30 * 24
        
        train = df.iloc[:train_end_idx]
        val = df.iloc[train_end_idx:val_end_idx]
        test = df.iloc[val_end_idx:]
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test