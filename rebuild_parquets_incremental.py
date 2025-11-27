"""
rebuild_parquets_incremental.py

Build C:XAU-USD parquets by quarter (3-month chunks), concatenating into final timeframe files.
This avoids memory exhaustion from processing entire 2020-2025 range at once.

Quarterly approach:
- Q1: Jan-Mar (typically lighter data)
- Q2: Apr-Jun (typically heavier)
- Q3: Jul-Sep (typically heavier)
- Q4: Oct-Dec (typically heavier)

Usage:
  python rebuild_parquets_incremental.py --symbol "C:XAU-USD" --start 2020 --end 2025 --out feature_store
"""

import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('rebuild_incremental')


def run_year_chunk(symbol: str, year: int, timeframes: list, out_dir: str) -> bool:
    """Run build_from_polygon_s3.py for a single year and return True if successful."""
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    
    cmd = [
        sys.executable, 'build_from_polygon_s3.py',
        '--symbol', symbol,
        '--start', start,
        '--end', end,
        '--timeframes'] + timeframes + [
        '--out', out_dir,
        '--delete-local'
    ]
    
    logger.info(f"Running {year}: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        logger.info(f"✓ Year {year} completed successfully")
        return True
    else:
        logger.warning(f"✗ Year {year} failed or was interrupted (exit code {result.returncode})")
        return False


def merge_yearly_parquets(symbol: str, years: list, timeframes: list, out_dir: str):
    """Merge yearly parquets into single per-timeframe files."""
    base = Path(out_dir) / symbol
    base.mkdir(parents=True, exist_ok=True)
    
    for tf in timeframes:
        yearly_files = []
        for year in years:
            # Each year run creates a file like C:XAU-USD_5T.parquet
            # We look for temp files or check if the main file was updated
            target = base / f"{symbol}_{tf}.parquet"
            if target.exists():
                yearly_files.append((year, target))
        
        if yearly_files:
            logger.info(f"Merging {tf} from {len(yearly_files)} year(s)")
            dfs = []
            for year, fpath in yearly_files:
                try:
                    df = pd.read_parquet(fpath)
                    logger.info(f"  Loaded {year}: {fpath.name} ({len(df)} rows)")
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"  Failed to load {year}: {e}")
            
            if dfs:
                # Concatenate and sort by timestamp
                merged = pd.concat(dfs, ignore_index=True)
                merged = merged.sort_values('timestamp').reset_index(drop=True)
                
                # Remove exact duplicates by timestamp
                merged = merged.drop_duplicates(subset=['timestamp'], keep='first')
                merged = merged.reset_index(drop=True)
                
                # Save back
                merged.to_parquet(str(base / f"{symbol}_{tf}.parquet"), index=False)
                logger.info(f"  Merged {len(merged)} total rows -> {base / f'{symbol}_{tf}.parquet'}")
            else:
                logger.warning(f"No data loaded for {tf}")
        else:
            logger.info(f"No files found for {tf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--start', type=int, required=True, help='Start year (e.g. 2020)')
    p.add_argument('--end', type=int, required=True, help='End year inclusive (e.g. 2025)')
    p.add_argument('--timeframes', nargs='+', default=['1T', '5T', '15T', '30T'])
    p.add_argument('--out', default='feature_store')
    args = p.parse_args()
    
    years = list(range(args.start, args.end + 1))
    logger.info(f"Will process years: {years}")
    
    successful_years = []
    for year in years:
        if run_year_chunk(args.symbol, year, args.timeframes, args.out):
            successful_years.append(year)
        # Continue even if one year fails, we can merge what we have
    
    if successful_years:
        logger.info(f"\nMerging {len(successful_years)} successful year(s): {successful_years}")
        merge_yearly_parquets(args.symbol, successful_years, args.timeframes, args.out)
        logger.info("✓ Merge complete")
    else:
        logger.error("No years completed successfully")


if __name__ == '__main__':
    main()
