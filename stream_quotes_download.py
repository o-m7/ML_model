"""
Streaming Quote Downloader - Memory Efficient
Aggregates tick data to OHLC bars directly without loading all quotes into memory
"""

import boto3
from botocore.config import Config
import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingQuotesDownloader:
    """Download quotes and aggregate efficiently"""
    
    def __init__(self):
        """Initialize S3"""
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
    
    def download_day_quotes(self, date_str):
        """Download single day's quotes"""
        date_obj = pd.to_datetime(date_str)
        year = date_obj.year
        month = f"{date_obj.month:02d}"
        
        object_key = f"global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz"
        
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
            with gzip.GzipFile(fileobj=response['Body']) as gzipfile:
                df = pd.read_csv(gzipfile)
            
            if 'participant_timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            
            # Compute basic metrics
            if 'bid_price' in df.columns and 'ask_price' in df.columns:
                df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
                df['spread'] = df['ask_price'] - df['bid_price']
            
            return df[['timestamp', 'mid_price', 'spread']].copy()
        
        except Exception as e:
            logger.debug(f"  Failed: {date_str}")
            return None
    
    def aggregate_quotes_to_ohlc(self, df, timeframe='5T'):
        """Convert quotes to OHLC bars"""
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        df = df.dropna(subset=['mid_price']).copy()
        if len(df) == 0:
            return pd.DataFrame()
        
        df_idx = df.set_index('timestamp')
        
        # OHLCV
        ohlc = df_idx['mid_price'].resample(timeframe).ohlc()
        volume = df_idx.resample(timeframe).size()
        spread_mean = df_idx['spread'].resample(timeframe).mean()
        
        result = ohlc.copy()
        result['volume'] = volume
        result['spread'] = spread_mean
        result = result.reset_index()
        
        return result
    
    def process_date_range(self, start_date='2020-01-01', end_date='2025-11-25'):
        """Download and aggregate all dates"""
        logger.info(f"\n{'='*70}")
        logger.info(f"STREAMING QUOTES DOWNLOAD & AGGREGATION")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"{'='*70}\n")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        
        # Storage for aggregated bars
        all_bars = {'5T': [], '15T': []}
        
        success = 0
        failed = 0
        
        for i, date in enumerate(date_range, 1):
            date_str = date.strftime("%Y-%m-%d")
            
            if i % 50 == 0:
                logger.info(f"[{i}/{len(date_range)}] {date_str} | Success: {success}, Failed: {failed}")
            
            # Download
            df_day = self.download_day_quotes(date_str)
            
            if df_day is not None and len(df_day) > 0:
                # Aggregate to timeframes
                for tf in ['5T', '15T']:
                    df_tf = self.aggregate_quotes_to_ohlc(df_day, tf)
                    if len(df_tf) > 0:
                        all_bars[tf].append(df_tf)
                
                success += 1
            else:
                failed += 1
        
        # Combine all bars
        logger.info(f"\n{'='*70}")
        logger.info(f"COMBINING BARS & SAVING")
        logger.info(f"{'='*70}")
        
        out_path = Path('feature_store') / 'C:XAU-USD' / 'quotes_real'
        out_path.mkdir(parents=True, exist_ok=True)
        
        for tf in ['5T', '15T']:
            if all_bars[tf]:
                df_combined = pd.concat(all_bars[tf], ignore_index=True)
                df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
                df_combined = df_combined.drop_duplicates(subset=['timestamp'])
                
                save_path = out_path / f'C:XAU-USD_{tf}_quotes.parquet'
                df_combined.to_parquet(save_path, compression='snappy', index=False)
                
                logger.info(f"✓ {tf}: {len(df_combined):,} bars → {save_path}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Success: {success}/{len(date_range)} days")


def main():
    try:
        downloader = StreamingQuotesDownloader()
        downloader.process_date_range(start_date='2020-01-01', end_date='2025-11-25')
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
