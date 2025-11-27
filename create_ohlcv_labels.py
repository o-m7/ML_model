"""
Create Labels for OHLCV Data
Generates binary classification labels for OHLCV candles
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def create_ohlcv_labels(
    df: pd.DataFrame,
    target_periods: int = 1,
    min_return_bps: float = 0.0
) -> pd.DataFrame:
    """Create binary labels for OHLCV data"""
    logger.info(f"Creating labels (target_periods={target_periods}, min_return={min_return_bps}bps)")
    
    df = df.copy()
    
    # Compute forward return
    df['close_fwd'] = df['close'].shift(-target_periods)
    df['return_fwd'] = ((df['close_fwd'] - df['close']) / df['close']) * 10000  # in bps
    
    # Create label
    df['label'] = (df['return_fwd'] > min_return_bps).astype(int)
    
    # Remove rows where we can't compute future return
    df = df.dropna(subset=['label']).reset_index(drop=True)
    
    logger.info(f"✓ Labels created: {len(df)} rows")
    logger.info(f"✓ Distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def label_all_ohlcv():
    """Create labels for all OHLCV timeframes"""
    data_dir = Path('feature_store/C:XAU-USD')
    timeframes = ['1T', '5T', '15T', '30T']
    
    logger.info("\n" + "="*70)
    logger.info("LABELING OHLCV DATA - ALL TIMEFRAMES")
    logger.info("="*70)
    
    for tf in timeframes:
        input_file = data_dir / f'C:XAU-USD_{tf}.parquet'
        output_file = data_dir / f'C:XAU-USD_{tf}_labeled.parquet'
        
        if not input_file.exists():
            logger.warning(f"File not found: {input_file}")
            continue
        
        logger.info(f"\nProcessing {tf}...")
        df = pd.read_parquet(input_file)
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Create labels
        df = create_ohlcv_labels(df, target_periods=1, min_return_bps=0.0)
        
        # Save
        df.to_parquet(output_file, compression='snappy', index=False)
        logger.info(f"✓ Saved: {output_file}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ OHLCV LABELING COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    label_all_ohlcv()
