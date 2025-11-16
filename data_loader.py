"""
Data loading and preprocessing module.

Handles loading OHLCV data for multiple symbols and timeframes,
with proper train/validation/test splits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def load_data_from_csv(
    file_path: str,
    symbol: str,
    timeframe: int
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.

    Expected CSV format:
    - timestamp (or date/datetime column)
    - open, high, low, close, volume

    Args:
        file_path: Path to CSV file
        symbol: Symbol name (for metadata)
        timeframe: Timeframe in minutes (for metadata)

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns
    """
    try:
        df = pd.read_csv(file_path)

        # Try to find timestamp column
        timestamp_cols = ['timestamp', 'time', 'date', 'datetime', 'Timestamp', 'Time', 'Date']
        ts_col = None
        for col in timestamp_cols:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None:
            # Assume first column is timestamp
            ts_col = df.columns[0]

        # Parse timestamp and set as index
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
        df.index.name = 'timestamp'

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Select only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Basic data validation
        df = df[(df['high'] >= df['low']) &
                (df['high'] >= df['close']) &
                (df['high'] >= df['open']) &
                (df['low'] <= df['close']) &
                (df['low'] <= df['open']) &
                (df['close'] > 0)]

        print(f"Loaded {len(df)} bars for {symbol} {timeframe}min: {df.index[0]} to {df.index[-1]}")

        return df

    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {e}")


def resample_data(df: pd.DataFrame, target_timeframe: int) -> pd.DataFrame:
    """
    Resample OHLCV data to target timeframe.

    Args:
        df: DataFrame with OHLCV data (any timeframe)
        target_timeframe: Target timeframe in minutes

    Returns:
        Resampled DataFrame
    """
    rule = f'{target_timeframe}min'

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets.

    NO shuffling - strict time-based split.

    Args:
        df: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing (final out-of-sample)

    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"  Train: {train_df.index[0]} to {train_df.index[-1]}")
    print(f"  Val:   {val_df.index[0]} to {val_df.index[-1]}")
    print(f"  Test:  {test_df.index[0]} to {test_df.index[-1]}")

    return train_df, val_df, test_df


def load_all_data(config) -> Dict[Tuple[str, int], Dict[str, pd.DataFrame]]:
    """
    Load data for all symbols and timeframes specified in config.

    Returns:
        Nested dict: {(symbol, timeframe): {'train': df, 'val': df, 'test': df}}
    """
    all_data = {}

    for symbol in config.data.symbols:
        for timeframe in config.data.timeframes:
            # Construct file path (adjust pattern to your data storage)
            # Example: data/XAUUSD_5.csv, data/XAUUSD_15.csv, etc.
            file_path = config.data.data_dir / f"{symbol}_{timeframe}.csv"

            if not file_path.exists():
                print(f"Warning: Data file not found: {file_path}")
                print(f"  Skipping {symbol} {timeframe}min")
                continue

            # Load data
            df = load_data_from_csv(str(file_path), symbol, timeframe)

            if len(df) < config.data.min_data_points:
                print(f"Warning: Insufficient data for {symbol} {timeframe}min ({len(df)} bars)")
                continue

            # Split data
            train_df, val_df, test_df = split_data(
                df,
                train_ratio=config.data.train_ratio,
                val_ratio=config.data.val_ratio,
                test_ratio=config.data.test_ratio
            )

            all_data[(symbol, timeframe)] = {
                'train': train_df,
                'val': val_df,
                'test': test_df,
                'full': df
            }

    print(f"\nLoaded data for {len(all_data)} symbol-timeframe pairs")

    return all_data


def create_sample_data(
    symbol: str,
    timeframe: int,
    n_bars: int = 20000,
    save_path: str = None
) -> pd.DataFrame:
    """
    Create synthetic OHLCV data for testing purposes.

    Simulates realistic price movements with trend and noise.

    Args:
        symbol: Symbol name
        timeframe: Timeframe in minutes
        n_bars: Number of bars to generate
        save_path: Optional path to save CSV

    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(42)

    # Start date
    start_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start_date, periods=n_bars, freq=f'{timeframe}min')

    # Initial price (realistic for XAUUSD/XAGUSD)
    if 'XAU' in symbol:
        initial_price = 1800.0
    elif 'XAG' in symbol:
        initial_price = 25.0
    else:
        initial_price = 100.0

    # Generate price with trend + noise
    trend = np.linspace(0, 0.1 * initial_price, n_bars)  # 10% trend
    noise = np.random.randn(n_bars).cumsum() * (initial_price * 0.001)  # Random walk
    cyclical = 0.05 * initial_price * np.sin(np.linspace(0, 10 * np.pi, n_bars))  # Cycles

    close_prices = initial_price + trend + noise + cyclical

    # Generate OHLC from close
    atr_pct = 0.01  # 1% average range
    df = pd.DataFrame({
        'close': close_prices
    }, index=dates)

    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(n_bars) * (df['close'] * atr_pct)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(n_bars) * (df['close'] * atr_pct)
    df['volume'] = np.random.randint(1000, 10000, n_bars)

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    df = df[['open', 'high', 'low', 'close', 'volume']]

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        print(f"Saved sample data to {save_path}")

    return df


if __name__ == "__main__":
    from config import get_default_config

    config = get_default_config()

    # Create sample data for testing
    print("Creating sample data...")

    for symbol in config.data.symbols:
        for timeframe in config.data.timeframes:
            file_path = config.data.data_dir / f"{symbol}_{timeframe}.csv"
            if not file_path.exists():
                df = create_sample_data(symbol, timeframe, n_bars=20000, save_path=str(file_path))
                print(f"Created {file_path}")

    # Test loading
    print("\n" + "="*60)
    print("Testing data loading...")
    all_data = load_all_data(config)

    for (symbol, tf), splits in all_data.items():
        print(f"\n{symbol} {tf}min:")
        print(f"  Train: {len(splits['train'])} bars")
        print(f"  Val:   {len(splits['val'])} bars")
        print(f"  Test:  {len(splits['test'])} bars")
