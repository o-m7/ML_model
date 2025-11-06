"""Data loading and preprocessing."""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import yaml


def load_symbol_data(
    symbol: str,
    timeframe: str,
    data_root: str = "feature_store",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol/timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe string (5T, 15T, etc.)
        data_root: Root directory containing data
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame with OHLCV data
    """
    path = Path(data_root) / symbol / f"{symbol}_{timeframe}.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    
    df = pd.read_parquet(path)
    
    # Ensure timestamp
    if 'timestamp' not in df.columns:
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise ValueError(f"No timestamp column found in {path}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Filter date range
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True)
        df = df[df['timestamp'] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date, utc=True)
        df = df[df['timestamp'] <= end_dt]
    
    return df.reset_index(drop=True)


class DataLoader:
    """Comprehensive data loading with configuration."""
    
    def __init__(self, config_path: str = "intraday_system/config/settings.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_root = self.config['data']['data_root']
        self.train_start = self.config['data']['train_start']
        self.train_end = self.config['data']['train_end']
        self.oos_months = self.config['data']['oos_months']
    
    def load_training_data(
        self,
        symbol: str,
        timeframe: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split into train and OOS.
        
        Returns:
            (train_df, oos_df)
        """
        # Load full data
        df = load_symbol_data(
            symbol,
            timeframe,
            self.data_root,
            self.train_start,
            self.train_end
        )
        
        # Calculate OOS split date
        from datetime import timedelta
        train_end_dt = pd.to_datetime(self.train_end, utc=True)
        oos_start_dt = train_end_dt - timedelta(days=self.oos_months * 30)
        
        # Split
        train_df = df[df['timestamp'] < oos_start_dt].copy()
        oos_df = df[df['timestamp'] >= oos_start_dt].copy()
        
        return train_df, oos_df
    
    def load_htf_data(
        self,
        symbol: str,
        htf_timeframe: str = "1D"
    ) -> Optional[pd.DataFrame]:
        """Load higher timeframe data (e.g., daily for 4H strategy)."""
        try:
            return load_symbol_data(
                symbol,
                htf_timeframe,
                self.data_root,
                self.train_start,
                self.train_end
            )
        except FileNotFoundError:
            return None

