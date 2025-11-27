"""
Download Real Quotes from Polygon S3 - Correct Implementation
Uses the proper path format from Polygon documentation
Downloads quotes from 2020-01-01 to 2025-11-25
"""

import boto3
from botocore.config import Config
import pandas as pd
import gzip
import io
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class PolygonS3QuotesDownloader:
    """Download real quote data from Polygon S3 using correct paths"""
    
    def __init__(self):
        """Initialize S3 client with credentials"""
        self.session = boto3.Session(
            aws_access_key_id='4937f95b-db8b-4d7e-8d54-756a82d4976e',
            aws_secret_access_key='o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1',
        )
        
        self.s3 = self.session.client(
            's3',
            endpoint_url='https://files.massive.com',
            config=Config(signature_version='s3v4'),
        )
        
        self.bucket_name = 'flatfiles'
        logger.info("✓ S3 client initialized")
    
    def download_quote_file(self, date_str):
        """Download single day's quote file"""
        date_obj = pd.to_datetime(date_str)
        year = date_obj.year
        month = f"{date_obj.month:02d}"
        day = date_obj.strftime("%Y-%m-%d")
        
        # Correct path format from Polygon
        object_key = f"global_forex/quotes_v1/{year}/{month}/{day}.csv.gz"
        
        try:
            # Download to memory
            response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
            
            # Decompress
            with gzip.GzipFile(fileobj=response['Body']) as gzipfile:
                df = pd.read_csv(gzipfile)
            
            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            
            return df
        
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.debug(f"Error downloading {date_str}: {e}")
            return None
    
    def download_quotes_range(self, start_date='2020-01-01', end_date='2025-11-25'):
        """Download all quotes in date range"""
        logger.info(f"\n{'='*70}")
        logger.info(f"DOWNLOADING REAL QUOTES FROM POLYGON S3")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"{'='*70}\n")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start_dt, end_dt, freq='D')
        
        all_quotes = []
        success = 0
        failed = 0
        
        for i, date in enumerate(date_range, 1):
            date_str = date.strftime("%Y-%m-%d")
            
            if i % 50 == 0 or i == 1:
                logger.info(f"[{i}/{len(date_range)}] {date_str}...")
            
            df = self.download_quote_file(date_str)
            
            if df is not None and len(df) > 0:
                all_quotes.append(df)
                success += 1
            else:
                failed += 1
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Success: {success}")
        logger.info(f"Failed: {failed}")
        
        if all_quotes:
            df_all = pd.concat(all_quotes, ignore_index=True)
            df_all = df_all.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"\n✓ Total quotes: {len(df_all):,}")
            logger.info(f"✓ Columns: {list(df_all.columns)}")
            logger.info(f"✓ Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
            
            return df_all
        else:
            logger.error("No quotes downloaded!")
            return None
    
    def aggregate_to_timeframe(self, df, timeframe='5T'):
        """Aggregate tick quotes to OHLC bars"""
        logger.info(f"Aggregating to {timeframe}...")
        
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute mid price if not present
        if 'mid_price' not in df.columns:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid_price'] = (df['bid'] + df['ask']) / 2
            elif 'price' in df.columns:
                df['mid_price'] = df['price']
        
        if 'mid_price' not in df.columns:
            logger.warning("No price data found!")
            return pd.DataFrame()
        
        # Set index for resampling
        df_idx = df.set_index('timestamp')
        
        # OHLCV from mid price
        ohlc = df_idx['mid_price'].resample(timeframe).ohlc()
        volume = df_idx.resample(timeframe).size()
        
        # Quote metrics
        agg_dict = {}
        if 'bid' in df_idx.columns:
            agg_dict['bid'] = ['first', 'last', 'min', 'max']
        if 'ask' in df_idx.columns:
            agg_dict['ask'] = ['first', 'last', 'min', 'max']
        if 'bid_size' in df_idx.columns:
            agg_dict['bid_size'] = ['sum', 'mean', 'max']
        if 'ask_size' in df_idx.columns:
            agg_dict['ask_size'] = ['sum', 'mean', 'max']
        
        # Aggregate
        quote_agg = df_idx.resample(timeframe).agg(agg_dict) if agg_dict else pd.DataFrame()
        
        # Combine
        result = ohlc.copy()
        result['volume'] = volume
        
        if len(quote_agg) > 0:
            quote_agg.columns = ['_'.join(col).strip('_') for col in quote_agg.columns.values]
            for col in quote_agg.columns:
                result[col] = quote_agg[col]
        
        result = result.reset_index()
        
        # Compute metrics
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
        """Save aggregated quotes to parquets"""
        out_path = Path(output_dir) / symbol / 'quotes'
        out_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SAVING QUOTE PARQUETS")
        logger.info(f"{'='*70}")
        
        # Save raw quotes first
        raw_path = out_path / f'{symbol}_quotes_raw.parquet'
        df.to_parquet(raw_path, compression='snappy', index=False)
        logger.info(f"✓ Saved raw: {raw_path}")
        
        # Aggregate and save timeframes
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
            start_date='2020-01-01',
            end_date='2025-11-25'
        )
        
        if df_quotes is not None and len(df_quotes) > 0:
            # Save to parquets
            output_path = downloader.save_quotes_parquets(df_quotes, symbol='C:XAU-USD')
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ REAL QUOTE DOWNLOAD & SAVE COMPLETE")
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
