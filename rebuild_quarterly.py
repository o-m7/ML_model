#!/usr/bin/env python3
"""
Rebuild C:XAU-USD parquets quarter-by-quarter with proper appending.
Each quarter's run overwrites temp files, so we append to the final parquets after each run.
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('rebuild_quarterly')

SYMBOL = "C:XAU-USD"
OUT_DIR = "feature_store"
TIMEFRAMES = ['5T', '15T', '30T']

# Generate all quarters: 2020 Q1-Q4, 2021 Q1-Q4, ..., 2025 Q1-Q4
QUARTERS = []
for year in range(2020, 2026):
    for q in range(1, 5):
        if year == 2025 and q > 3:  # Only Q1-Q3 for 2025
            continue
        q_months = {1: ('01-01', '03-31'), 2: ('04-01', '06-30'), 3: ('07-01', '09-30'), 4: ('10-01', '12-31')}
        start_m, end_m = q_months[q]
        QUARTERS.append((year, q, f"{year}-{start_m}", f"{year}-{end_m}"))

logger.info(f"Will process {len(QUARTERS)} quarters: 2020 Q1 → 2025 Q3")

def run_quarter(year: int, q: int, start: str, end: str) -> bool:
    """Run build_from_polygon_s3 for one quarter."""
    cmd = [
        sys.executable, 'build_from_polygon_s3.py',
        '--symbol', SYMBOL,
        '--start', start,
        '--end', end,
        '--timeframes'] + TIMEFRAMES + [
        '--out', OUT_DIR,
        '--delete-local'
    ]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {year} Q{q} ({start} to {end})")
    logger.info(f"{'='*70}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        logger.info(f"✓ {year} Q{q} completed")
        return True
    else:
        logger.error(f"✗ {year} Q{q} failed (exit code {result.returncode})")
        return False


def append_quarterly_data(year: int, q: int):
    """
    After a quarter completes, the build script creates parquets.
    We append them to permanent storage and clear the temp files.
    """
    base = Path(OUT_DIR) / SYMBOL
    
    for tf in TIMEFRAMES:
        temp_file = base / f"{SYMBOL}_{tf}.parquet"
        final_file = base / f"{SYMBOL}_{tf}_full.parquet"
        
        if not temp_file.exists():
            logger.warning(f"  {tf}: No temp file found")
            continue
        
        try:
            df_new = pd.read_parquet(temp_file)
            logger.info(f"  {tf}: Read {len(df_new)} rows from temp")
            
            if final_file.exists():
                df_existing = pd.read_parquet(final_file)
                df_merged = pd.concat([df_existing, df_new], ignore_index=True)
                df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
                # Remove exact duplicates by timestamp
                df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='first')
                logger.info(f"  {tf}: Merged {len(df_existing)} existing + {len(df_new)} new = {len(df_merged)} total")
            else:
                df_merged = df_new.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"  {tf}: First batch: {len(df_merged)} rows")
            
            df_merged.to_parquet(str(final_file), index=False)
        except Exception as e:
            logger.error(f"  {tf}: Failed to append - {e}")


def main():
    successful = []
    failed = []
    
    for year, q, start, end in QUARTERS:
        if run_quarter(year, q, start, end):
            append_quarterly_data(year, q)
            successful.append(f"{year}Q{q}")
        else:
            failed.append(f"{year}Q{q}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY:")
    logger.info(f"  Successful: {len(successful)} - {', '.join(successful)}")
    if failed:
        logger.info(f"  Failed: {len(failed)} - {', '.join(failed)}")
    
    # Rename final files to standard names
    base = Path(OUT_DIR) / SYMBOL
    for tf in TIMEFRAMES:
        final_full = base / f"{SYMBOL}_{tf}_full.parquet"
        final_std = base / f"{SYMBOL}_{tf}.parquet"
        if final_full.exists():
            final_full.rename(final_std)
            df = pd.read_parquet(final_std)
            logger.info(f"✓ {final_std.name}: {len(df)} total rows, {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == '__main__':
    main()
