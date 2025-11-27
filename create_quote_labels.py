"""
Create Labels for Quote Data
Generates binary classification labels (1=up, 0=down) based on future returns
using a simple forward-looking approach with no look-ahead bias.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def create_labels_from_quotes(
    df: pd.DataFrame,
    target_periods: int = 1,
    min_return_bps: float = 0.0
) -> pd.DataFrame:
    """
    Create binary labels based on future returns (NO look-ahead bias).
    
    Logic:
    - For each bar, compute return of next N bars
    - If return > min_return_bps: label = 1 (up)
    - Otherwise: label = 0 (down)
    
    Args:
        df: Dataframe with 'close' column
        target_periods: Number of periods ahead to compute return
        min_return_bps: Minimum return in basis points to be labeled as "up"
    
    Returns:
        Dataframe with 'label' column added
    """
    logger.info("\n" + "="*70)
    logger.info(f"CREATING LABELS (target_periods={target_periods}, min_return={min_return_bps}bps)")
    logger.info("="*70)
    
    df = df.copy()
    
    # Compute forward return (future close - current close)
    df['close_fwd'] = df['close'].shift(-target_periods)
    df['return_fwd'] = ((df['close_fwd'] - df['close']) / df['close']) * 10000  # in bps
    
    # Create label: 1 if future return > threshold, 0 otherwise
    df['label'] = (df['return_fwd'] > min_return_bps).astype(int)
    
    # Remove rows where we can't compute future return (last N rows)
    df = df.dropna(subset=['label']).reset_index(drop=True)
    
    logger.info(f"✓ Labels created")
    logger.info(f"✓ Rows with valid labels: {len(df)}")
    logger.info(f"✓ Label distribution: {df['label'].value_counts().to_dict()}")
    logger.info(f"✓ Ratio: {df['label'].sum() / len(df):.2%} up moves")
    
    return df


def label_all_timeframes():
    """Create labels for all available timeframes"""
    data_dir = Path('feature_store/C:XAU-USD/quotes')
    timeframes = ['1T', '5T', '15T', '30T']
    
    logger.info("\n" + "="*70)
    logger.info("LABELING ALL TIMEFRAMES")
    logger.info("="*70)
    
    for tf in timeframes:
        input_file = data_dir / f'C:XAU-USD_{tf}_quotes_advanced.parquet'
        output_file = data_dir / f'C:XAU-USD_{tf}_quotes_labeled.parquet'
        
        if not input_file.exists():
            logger.warning(f"File not found: {input_file}")
            continue
        
        logger.info(f"\nProcessing {tf}...")
        df = pd.read_parquet(input_file)
        
        # Create labels
        df = create_labels_from_quotes(df, target_periods=1, min_return_bps=0.0)
        
        # Save
        df.to_parquet(output_file, compression='snappy', index=False)
        logger.info(f"✓ Saved: {output_file}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ LABELING COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    label_all_timeframes()
