"""
Download Real Quotes from Polygon S3
Uses credentials from .env to fetch actual bid/ask quote data
Aggregates to multiple timeframes and computes features
"""

import os
import boto3
from botocore.config import Config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import gzip
import io
from dotenv import load_dotenv

# Load environment
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class PolygonS3QuotesDownloader:
    """Download real quote data from Polygon S3"""
    
    def __init__(self):
        """Initialize S3 client with credentials from .env"""
        self.access_key = os.getenv('POLYGON_S3_ACCESS_KEY')
        self.secret_key = os.getenv('Secret_Access_Key')
        self.endpoint_url = os.getenv('POLYGON_S3_SECRET_KEY')
        self.bucket = os.getenv('Bucket', 'flatfiles')
        
        if not all([self.access_key, self.secret_key, self.endpoint_url]):
            raise ValueError("Missing S3 credentials in .env")
        
        logger.info(f"S3 Credentials loaded:")
        logger.info(f"  Bucket: {self.bucket}")
        logger.info(f"  Endpoint: {self.endpoint_url}")
        
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            endpoint_url=self.endpoint_url,
            config=Config(signature_version='s3v4', max_pool_connections=10),
        )
        
        logger.info("✓ S3 client initialized")
    
    def list_available_quotes(self, symbol='C:XAU-USD', start_date='2020-01-01', end_date='2025-11-25'):
        """List available quote files on S3"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"CHECKING AVAILABLE QUOTES: {symbol}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"{'='*70}")
        
        # Generate all dates
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        
        available_dates = []
        checked = 0
        found = 0
        
        for date in date_range:
            year = date.year
            month = f"{date.month:02d}"
            day = date.strftime("%Y-%m-%d")
            
            # Path to quote file
            key = f"flatfiles/global_forex/quotes_v1/{year}/{month}/{day}.csv.gz"
            
            checked += 1
            if checked % 50 == 0:
                logger.info(f"  Checked {checked} dates, found {found} files...")
            
            try:
                self.s3.head_object(Bucket=self.bucket, Key=key)
                available_dates.append(day)
                found += 1
            except:
                pass
        
        logger.info(f"\n✓ Total checked: {checked}")
        logger.info(f"✓ Files found: {found}")
        
        if available_dates:
            logger.info(f"✓ Date range found: {available_dates[0]} to {available_dates[-1]}")
        
        return available_dates
    
    def download_quote_file(self, date_str):
        """Download single day's quote file"""
        date_obj = pd.to_datetime(date_str)
        year = date_obj.year
        month = f"{date_obj.month:02d}"
        
        key = f"flatfiles/global_forex/quotes_v1/{year}/{month}/{date_str}.csv.gz"
        
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            
            # Decompress and read
            with gzip.GzipFile(fileobj=response['Body']) as gzipfile:
                df = pd.read_csv(gzipfile)
            
            # Parse timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            if 'quote_at' in df.columns:
                df['quote_at'] = pd.to_datetime(df['quote_at'], unit='ns')
            
            return df
        
        except Exception as e:
            logger.debug(f"  Failed to download {date_str}: {e}")
            return None
    
    def download_quotes_range(self, symbol, start_date='2020-01-01', end_date='2025-11-25'):
        """Download all quotes in date range"""
        logger.info(f"\n{'='*70}")
        logger.info(f"DOWNLOADING QUOTES: {symbol}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"{'='*70}\n")
        
        # First check what's available
        available_dates = self.list_available_quotes(symbol, start_date, end_date)
        
        if not available_dates:
            logger.error("✗ No quote files found on S3!")
            return None
        
        # Download all available files
        all_quotes = []
        failed = 0
        success = 0
        
        for i, date_str in enumerate(available_dates, 1):
            logger.info(f"[{i}/{len(available_dates)}] Downloading {date_str}...")
            
            df = self.download_quote_file(date_str)
            if df is not None and len(df) > 0:
                all_quotes.append(df)
                success += 1
                logger.info(f"  ✓ {len(df):,} quotes")
            else:
                failed += 1
        
        # Combine all
        logger.info(f"\n{'='*70}")
        logger.info(f"COMBINING {success} FILES")
        logger.info(f"{'='*70}")
        
        if all_quotes:
            df_all = pd.concat(all_quotes, ignore_index=True)
            df_all = df_all.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"✓ Total quotes: {len(df_all):,}")
            logger.info(f"✓ Columns: {list(df_all.columns)}")
            logger.info(f"✓ Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
            
            return df_all
        else:
            logger.error("Failed to download any quotes")
            return None
    
    def aggregate_to_timeframe(self, df, timeframe='5T'):
        """Aggregate tick data to OHLC bars with quote features"""
        logger.info(f"Aggregating to {timeframe}...")
        
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Use mid_price if available, else compute from bid/ask
        if 'mid_price' not in df.columns and 'bid' in df.columns and 'ask' in df.columns:
            df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        if 'mid_price' not in df.columns:
            logger.warning("No mid_price or bid/ask columns found")
            return pd.DataFrame()
        
        # Set index for resampling
        df_idx = df.set_index('timestamp')
        
        # OHLCV
        ohlc = df_idx['mid_price'].resample(timeframe).ohlc()
        volume = df_idx.resample(timeframe).size()
        
        # Quote metrics
        agg_funcs = {}
        if 'bid' in df_idx.columns:
            agg_funcs['bid'] = ['first', 'last', 'min', 'max']
        if 'ask' in df_idx.columns:
            agg_funcs['ask'] = ['first', 'last', 'min', 'max']
        if 'bid_size' in df_idx.columns:
            agg_funcs['bid_size'] = ['sum', 'mean', 'max']
        if 'ask_size' in df_idx.columns:
            agg_funcs['ask_size'] = ['sum', 'mean', 'max']
        
        quote_agg = df_idx.resample(timeframe).agg(agg_funcs) if agg_funcs else pd.DataFrame()
        
        # Combine
        result = ohlc.copy()
        result['volume'] = volume
        
        if len(quote_agg) > 0:
            quote_agg.columns = ['_'.join(col).strip('_') for col in quote_agg.columns.values]
            for col in quote_agg.columns:
                result[col] = quote_agg[col]
        
        result = result.reset_index()
        
        # Compute spread and pressure metrics
        if 'ask_first' in result.columns and 'bid_first' in result.columns:
            result['spread'] = result['ask_first'] - result['bid_first']
            result['spread_pct'] = (result['spread'] / result['open']) * 10000
        
        if 'bid_size_sum' in result.columns and 'ask_size_sum' in result.columns:
            total_size = result['bid_size_sum'] + result['ask_size_sum']
            total_size = total_size.replace(0, 1)
            result['buy_pressure'] = result['bid_size_sum'] / total_size
            result['sell_pressure'] = result['ask_size_sum'] / total_size
        
        logger.info(f"✓ Aggregated {timeframe}: {len(result)} bars, {len(result.columns)} features")
        return result
    
    def save_quotes_parquets(self, df, symbol='C:XAU-USD', output_dir='feature_store'):
        """Save quotes to timeframe-specific parquets"""
        out_path = Path(output_dir) / symbol / 'quotes'
        out_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SAVING QUOTE PARQUETS")
        logger.info(f"{'='*70}")
        
        timeframes = ['1T', '5T', '15T', '30T']
        
        for tf in timeframes:
            df_tf = self.aggregate_to_timeframe(df, tf)
            
            if len(df_tf) > 0:
                save_path = out_path / f'{symbol}_{tf}_quotes.parquet'
                df_tf.to_parquet(save_path, compression='snappy', index=False)
                logger.info(f"✓ Saved: {save_path}")
        
        logger.info(f"\n✓ All parquets saved to {out_path}")
        return str(out_path)


def main():
    try:
        downloader = PolygonS3QuotesDownloader()
        
        # Download quotes
        df_quotes = downloader.download_quotes_range(
            symbol='C:XAU-USD',
            start_date='2020-01-01',
            end_date='2025-11-25'
        )
        
        if df_quotes is not None and len(df_quotes) > 0:
            # Save to parquets
            output_path = downloader.save_quotes_parquets(df_quotes, symbol='C:XAU-USD')
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ REAL QUOTE DOWNLOAD COMPLETE")
            logger.info(f"{'='*70}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Total quotes: {len(df_quotes):,}")
            logger.info(f"Date range: {df_quotes['timestamp'].min()} to {df_quotes['timestamp'].max()}")
        else:
            logger.error("Failed to download quotes")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
