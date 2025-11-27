"""
Download Real Quotes from Polygon S3 - Optimized with Parallel Downloads
Downloads quotes from 2020-01-01 to 2025-11-25 efficiently
"""

import boto3
from botocore.config import Config
import pandas as pd
import gzip
from pathlib import Path
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedQuotesDownloader:
    """Download quotes with parallel downloads"""
    
    def __init__(self, max_workers=5):
        """Initialize S3 client"""
        self.session = boto3.Session(
            aws_access_key_id='4937f95b-db8b-4d7e-8d54-756a82d4976e',
            aws_secret_access_key='o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1',
        )
        
        self.s3 = self.session.client(
            's3',
            endpoint_url='https://files.massive.com',
            config=Config(signature_version='s3v4', max_pool_connections=max_workers),
        )
        
        self.bucket_name = 'flatfiles'
        self.max_workers = max_workers
        logger.info(f"✓ S3 client initialized (workers={max_workers})")
    
    def download_single_quote(self, date_str):
        """Download single day's quotes"""
        date_obj = pd.to_datetime(date_str)
        year = date_obj.year
        month = f"{date_obj.month:02d}"
        
        object_key = f"global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz"
        
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
            with gzip.GzipFile(fileobj=response['Body']) as gzipfile:
                df = pd.read_csv(gzipfile)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            
            return (date_str, df)
        
        except Exception as e:
            return (date_str, None)
    
    def download_quotes_parallel(self, start_date='2020-01-01', end_date='2025-11-25'):
        """Download quotes with parallel workers"""
        logger.info(f"\n{'='*70}")
        logger.info(f"DOWNLOADING REAL QUOTES (PARALLEL)")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Workers: {self.max_workers}")
        logger.info(f"{'='*70}\n")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        date_list = [d.strftime("%Y-%m-%d") for d in date_range]
        
        logger.info(f"Total dates to download: {len(date_list)}")
        
        all_quotes = []
        success = 0
        failed = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_quote, date): date for date in date_list}
            
            for i, future in enumerate(as_completed(futures), 1):
                date_str, df = future.result()
                
                if df is not None and len(df) > 0:
                    all_quotes.append(df)
                    success += 1
                else:
                    failed += 1
                
                if i % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    remaining = (len(date_list) - i) / rate if rate > 0 else 0
                    logger.info(f"  Progress: {i}/{len(date_list)} | Success: {success} | "
                              f"Rate: {rate:.1f} jobs/sec | Est. remaining: {remaining:.0f}s")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DOWNLOAD SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Success: {success}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total files: {len(date_list)}")
        logger.info(f"Time: {time.time() - start_time:.1f}s")
        
        if all_quotes:
            logger.info(f"\nCombining {len(all_quotes)} files...")
            df_all = pd.concat(all_quotes, ignore_index=True)
            df_all = df_all.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"✓ Total quotes: {len(df_all):,}")
            logger.info(f"✓ Columns: {list(df_all.columns)}")
            logger.info(f"✓ Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
            
            return df_all
        else:
            logger.error("No quotes downloaded!")
            return None
    
    def aggregate_to_timeframe(self, df, timeframe='5T'):
        """Aggregate quotes to OHLC"""
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute mid price
        if 'mid_price' not in df.columns:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid_price'] = (df['bid'] + df['ask']) / 2
            elif 'price' in df.columns:
                df['mid_price'] = df['price']
        
        if 'mid_price' not in df.columns:
            logger.warning("No price data!")
            return pd.DataFrame()
        
        df_idx = df.set_index('timestamp')
        
        # OHLCV
        ohlc = df_idx['mid_price'].resample(timeframe).ohlc()
        volume = df_idx.resample(timeframe).size()
        
        # Quote agg
        agg_dict = {}
        if 'bid' in df_idx.columns:
            agg_dict['bid'] = ['first', 'last']
        if 'ask' in df_idx.columns:
            agg_dict['ask'] = ['first', 'last']
        if 'bid_size' in df_idx.columns:
            agg_dict['bid_size'] = ['sum', 'mean']
        if 'ask_size' in df_idx.columns:
            agg_dict['ask_size'] = ['sum', 'mean']
        
        quote_agg = df_idx.resample(timeframe).agg(agg_dict) if agg_dict else pd.DataFrame()
        
        result = ohlc.copy()
        result['volume'] = volume
        
        if len(quote_agg) > 0:
            quote_agg.columns = ['_'.join(col).strip('_') for col in quote_agg.columns.values]
            for col in quote_agg.columns:
                result[col] = quote_agg[col]
        
        result = result.reset_index()
        
        # Metrics
        if 'ask_first' in result.columns and 'bid_first' in result.columns:
            result['spread'] = result['ask_first'] - result['bid_first']
            result['spread_pct'] = (result['spread'] / result['open']) * 10000
        
        if 'bid_size_sum' in result.columns and 'ask_size_sum' in result.columns:
            total = result['bid_size_sum'] + result['ask_size_sum']
            total = total.replace(0, 1)
            result['buy_pressure'] = result['bid_size_sum'] / total
        
        logger.info(f"✓ {timeframe}: {len(result)} bars")
        return result
    
    def save_quotes_parquets(self, df, symbol='C:XAU-USD'):
        """Save to parquets"""
        out_path = Path('feature_store') / symbol / 'quotes'
        out_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SAVING QUOTE PARQUETS")
        logger.info(f"{'='*70}")
        
        # Raw
        raw_path = out_path / f'{symbol}_quotes_raw.parquet'
        df.to_parquet(raw_path, compression='snappy', index=False)
        logger.info(f"✓ Raw: {raw_path}")
        
        # Timeframes
        for tf in ['1T', '5T', '15T', '30T']:
            df_tf = self.aggregate_to_timeframe(df, tf)
            if len(df_tf) > 0:
                path = out_path / f'{symbol}_{tf}_quotes.parquet'
                df_tf.to_parquet(path, compression='snappy', index=False)
                logger.info(f"✓ Saved: {path}")
        
        return str(out_path)


def main():
    try:
        downloader = OptimizedQuotesDownloader(max_workers=5)
        
        df_quotes = downloader.download_quotes_parallel(
            start_date='2020-01-01',
            end_date='2025-11-25'
        )
        
        if df_quotes is not None and len(df_quotes) > 0:
            output_path = downloader.save_quotes_parquets(df_quotes)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ REAL QUOTES DOWNLOAD COMPLETE")
            logger.info(f"{'='*70}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Total: {len(df_quotes):,} quotes")
        else:
            logger.error("Failed!")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
