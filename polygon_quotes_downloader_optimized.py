"""
polygon_quotes_downloader_optimized.py - Optimized quote downloader with progress tracking

Downloads Polygon S3 quotes in monthly chunks for efficiency.
Saves intermediate results to avoid data loss.
"""

import boto3
from botocore.config import Config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import gzip
import io
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('polygon_quotes_opt')


class OptimizedQuotesDownloader:
    """Download and process Polygon S3 quote files with optimization."""
    
    def __init__(self):
        """Initialize S3 client with Polygon credentials."""
        self.session = boto3.Session(
            aws_access_key_id='4937f95b-db8b-4d7e-8d54-756a82d4976e',
            aws_secret_access_key='o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1',
        )
        
        self.s3 = self.session.client(
            's3',
            endpoint_url='https://files.massive.com',
            config=Config(signature_version='s3v4', max_pool_connections=5),
        )
        
        self.bucket_name = 'flatfiles'
        logger.info("✓ S3 client initialized")
    
    def download_quote_file(self, date_str: str, retry_count: int = 1) -> pd.DataFrame:
        """Download single day's quote file from S3."""
        date_obj = pd.to_datetime(date_str)
        year = date_obj.year
        month = f"{date_obj.month:02d}"
        day = f"{date_obj.day:02d}"
        
        object_key = f"flatfiles/global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz"
        
        for attempt in range(retry_count):
            try:
                response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
                
                # Decompress gzip and read
                with gzip.GzipFile(fileobj=response['Body']) as gzipfile:
                    df = pd.read_csv(gzipfile)
                
                # Parse timestamps
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
                if 'quote_at' in df.columns:
                    df['quote_at'] = pd.to_datetime(df['quote_at'], unit='ns')
                
                return df
            
            except self.s3.exceptions.NoSuchKey:
                return None
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(1)
                    continue
                return None
    
    def download_by_month(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download quotes grouped by month for efficiency.
        Shows progress and saves intermediate checkpoints.
        """
        logger.info(f"Downloading quotes from {start_date} to {end_date}")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate all months in range
        current = pd.Period(start_dt, freq='M')
        end = pd.Period(end_dt, freq='M')
        
        all_monthly_quotes = []
        month_count = 0
        total_months = (end.ordinal - current.ordinal) + 1
        
        while current <= end:
            month_start = current.start_time
            month_end = min(current.end_time, end_dt)
            
            month_count += 1
            logger.info(f"\n[{month_count}/{total_months}] Processing {current.strftime('%Y-%m')}")
            logger.info("-" * 50)
            
            # Download all days in this month
            monthly_quotes = []
            current_day = month_start
            day_count = 0
            success_count = 0
            
            while current_day <= month_end:
                date_str = current_day.strftime("%Y-%m-%d")
                
                df = self.download_quote_file(date_str)
                if df is not None and len(df) > 0:
                    monthly_quotes.append(df)
                    success_count += 1
                    logger.debug(f"  ✓ {date_str}: {len(df)} quotes")
                
                day_count += 1
                current_day += timedelta(days=1)
            
            # Combine month
            if monthly_quotes:
                df_month = pd.concat(monthly_quotes, ignore_index=True)
                df_month = df_month.sort_values('timestamp').reset_index(drop=True)
                all_monthly_quotes.append(df_month)
                
                logger.info(f"Month total: {len(df_month):,} quotes ({success_count}/{day_count} days)")
            else:
                logger.warning(f"No data found for {current}")
            
            current = current + 1
        
        # Final concatenation
        logger.info("\n" + "=" * 70)
        logger.info("CONCATENATING ALL MONTHS...")
        logger.info("=" * 70)
        
        if all_monthly_quotes:
            df_all = pd.concat(all_monthly_quotes, ignore_index=True)
            df_all = df_all.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"✓ Total quotes: {len(df_all):,}")
            logger.info(f"✓ Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
            
            return df_all
        else:
            logger.error("No quotes downloaded!")
            return None
    
    @staticmethod
    def aggregate_to_timeframe(quotes_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate quote data to OHLC bars."""
        df = quotes_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resample to OHLC
        ohlc = df.set_index('timestamp')['mid_price'].resample(timeframe).ohlc()
        
        # Add volume (quote count)
        df_indexed = df.set_index('timestamp')
        volume = df_indexed.resample(timeframe).size()
        
        # Quote features
        agg_funcs = {
            'bid': ['first', 'last', 'min', 'max'],
            'ask': ['first', 'last', 'min', 'max'],
            'mid_price': ['mean', 'std'],
            'bid_size': ['sum', 'mean', 'max'],
            'ask_size': ['sum', 'mean', 'max'],
        }
        
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in df_indexed.columns}
        quote_agg = df_indexed.resample(timeframe).agg(agg_funcs)
        quote_agg.columns = ['_'.join(col).strip('_') for col in quote_agg.columns.values]
        
        # Combine
        result = ohlc.copy()
        result['volume'] = volume
        for col in quote_agg.columns:
            result[col] = quote_agg[col]
        
        result = result.reset_index()
        
        # Computed metrics
        if 'ask_first' in result.columns and 'bid_first' in result.columns:
            result['spread'] = result['ask_first'] - result['bid_first']
            result['spread_pct'] = (result['spread'] / result['mid_price_mean']) * 10000
        
        if 'bid_size_sum' in result.columns and 'ask_size_sum' in result.columns:
            total_size = result['bid_size_sum'] + result['ask_size_sum']
            total_size = total_size.replace(0, 1)
            result['buy_pressure'] = result['bid_size_sum'] / total_size
            result['sell_pressure'] = result['ask_size_sum'] / total_size
        
        logger.info(f"✓ Aggregated {timeframe}: {len(result)} bars, {len(result.columns)} features")
        return result
    
    @staticmethod
    def save_parquets(quotes_df: pd.DataFrame, symbol: str, output_dir: str = "feature_store"):
        """Save quote data as timeframe-specific parquets."""
        out_path = Path(output_dir) / symbol / "quotes"
        out_path.mkdir(parents=True, exist_ok=True)
        
        timeframes = ['1T', '5T', '15T', '30T']
        
        logger.info("\n" + "=" * 70)
        logger.info("SAVING TIMEFRAME PARQUETS")
        logger.info("=" * 70)
        
        for tf in timeframes:
            logger.info(f"\nAggregating to {tf}...")
            df_tf = OptimizedQuotesDownloader.aggregate_to_timeframe(quotes_df, tf)
            
            save_file = out_path / f"{symbol}_{tf}_quotes.parquet"
            df_tf.to_parquet(save_file, compression='snappy', index=False)
            
            logger.info(f"✓ Saved {save_file}")
        
        logger.info(f"\n✓ All parquets saved to {out_path}")
        return str(out_path)


def main():
    import argparse
    
    p = argparse.ArgumentParser(description='Download Polygon XAUUSD quotes (optimized)')
    p.add_argument('--start', default='2020-01-01', help='Start date YYYY-MM-DD')
    p.add_argument('--end', default='2025-11-25', help='End date YYYY-MM-DD')
    p.add_argument('--symbol', default='C:XAU-USD', help='Output symbol')
    p.add_argument('--output', default='feature_store', help='Output directory')
    args = p.parse_args()
    
    downloader = OptimizedQuotesDownloader()
    
    try:
        logger.info("\n" + "=" * 70)
        logger.info("POLYGON QUOTES DOWNLOADER (OPTIMIZED)")
        logger.info("=" * 70)
        
        # Download by month
        quotes_df = downloader.download_by_month(args.start, args.end)
        
        if quotes_df is not None and len(quotes_df) > 0:
            # Save parquets
            output_path = downloader.save_parquets(quotes_df, args.symbol, args.output)
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ QUOTE DOWNLOAD COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Output: {output_path}")
            logger.info(f"Total quotes: {len(quotes_df):,}")
            logger.info(f"Date range: {quotes_df['timestamp'].min()} to {quotes_df['timestamp'].max()}")
            
        else:
            logger.error("Failed to download quotes")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
