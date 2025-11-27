"""
Download Real Quotes from Polygon S3 - Month by Month with Feature Computation
Processes data efficiently without loading entire date range at once
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


class MonthlyQuotesProcessor:
    """Download and process quotes month by month"""
    
    def __init__(self):
        """Initialize S3 client"""
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
    
    def download_month_quotes(self, year, month):
        """Download all quotes for a month"""
        month_str = f"{month:02d}"
        
        # Generate all dates in month
        start = pd.to_datetime(f"{year}-{month_str}-01")
        if month == 12:
            end = pd.to_datetime(f"{year+1}-01-01") - pd.Timedelta(days=1)
        else:
            next_month = f"{month+1:02d}"
            end = pd.to_datetime(f"{year}-{next_month}-01") - pd.Timedelta(days=1)
        
        date_range = pd.date_range(start, end, freq='D')
        
        all_quotes = []
        success = 0
        failed = 0
        
        logger.info(f"  Downloading {len(date_range)} days...")
        
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            object_key = f"global_forex/quotes_v1/{year}/{month_str}/{date_str}.csv.gz"
            
            try:
                response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
                with gzip.GzipFile(fileobj=response['Body']) as gzipfile:
                    df = pd.read_csv(gzipfile)
                
                if 'participant_timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
                
                # Compute mid price
                if 'bid_price' in df.columns and 'ask_price' in df.columns:
                    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
                    df['spread'] = df['ask_price'] - df['bid_price']
                
                all_quotes.append(df)
                success += 1
                
            except Exception as e:
                failed += 1
        
        if all_quotes:
            df_month = pd.concat(all_quotes, ignore_index=True)
            df_month = df_month.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"  ✓ Downloaded {len(df_month):,} quotes ({success} days)")
            return df_month
        else:
            logger.warning(f"  ✗ No data found for {year}-{month_str}")
            return None
    
    def aggregate_month_to_timeframes(self, df_month, year, month):
        """Aggregate month quotes to multiple timeframes"""
        if df_month is None or len(df_month) == 0:
            return {}
        
        result_dfs = {}
        
        for timeframe in ['1T', '5T', '15T', '30T']:
            df = df_month.copy()
            df = df.sort_values('timestamp').reset_index(drop=True)
            df_idx = df.set_index('timestamp')
            
            # OHLCV
            if 'mid_price' in df.columns:
                ohlc = df_idx['mid_price'].resample(timeframe).ohlc()
                volume = df_idx.resample(timeframe).size()
                
                # Spread metrics
                spread_mean = df_idx['spread'].resample(timeframe).mean() if 'spread' in df_idx.columns else None
                
                # Bid/Ask sizes
                bid_size = df_idx['bid_size'].resample(timeframe).sum() if 'bid_size' in df_idx.columns else None
                ask_size = df_idx['ask_size'].resample(timeframe).sum() if 'ask_size' in df_idx.columns else None
                
                # Combine
                result = ohlc.copy()
                result['volume'] = volume
                
                if spread_mean is not None:
                    result['spread'] = spread_mean
                
                if bid_size is not None and ask_size is not None:
                    result['bid_size'] = bid_size
                    result['ask_size'] = ask_size
                    total = bid_size + ask_size
                    total = total.replace(0, 1)
                    result['buy_pressure'] = bid_size / total
                
                result = result.reset_index()
                result_dfs[timeframe] = result
        
        return result_dfs
    
    def process_and_save(self, start_date='2020-01-01', end_date='2025-11-25'):
        """Process all months and save to parquets"""
        logger.info(f"\n{'='*70}")
        logger.info(f"DOWNLOADING REAL QUOTES MONTH BY MONTH")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"{'='*70}\n")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate year/month pairs
        current = pd.Period(start_dt, freq='M')
        end = pd.Period(end_dt, freq='M')
        
        out_path = Path('feature_store') / 'C:XAU-USD' / 'quotes_real'
        out_path.mkdir(parents=True, exist_ok=True)
        
        all_aggregates = {'1T': [], '5T': [], '15T': [], '30T': []}
        
        month_count = 0
        total_months = (end.ordinal - current.ordinal) + 1
        
        while current <= end:
            month_count += 1
            year = current.year
            month = current.month
            
            logger.info(f"[{month_count}/{total_months}] {year}-{month:02d}")
            
            # Download month
            df_month = self.download_month_quotes(year, month)
            
            if df_month is not None:
                # Aggregate to timeframes
                agg_dfs = self.aggregate_month_to_timeframes(df_month, year, month)
                
                for tf, df_tf in agg_dfs.items():
                    if len(df_tf) > 0:
                        all_aggregates[tf].append(df_tf)
            
            current = current + 1
        
        # Save final aggregates
        logger.info(f"\n{'='*70}")
        logger.info(f"SAVING AGGREGATED PARQUETS")
        logger.info(f"{'='*70}")
        
        for tf in ['5T', '15T']:  # Focus on 5T and 15T
            if all_aggregates[tf]:
                df_final = pd.concat(all_aggregates[tf], ignore_index=True)
                df_final = df_final.sort_values('timestamp').reset_index(drop=True)
                
                save_path = out_path / f'C:XAU-USD_{tf}_quotes.parquet'
                df_final.to_parquet(save_path, compression='snappy', index=False)
                
                logger.info(f"✓ {tf}: {len(df_final):,} bars saved to {save_path}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ REAL QUOTES PROCESSING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Output: {out_path}")


def main():
    try:
        processor = MonthlyQuotesProcessor()
        processor.process_and_save(start_date='2020-01-01', end_date='2025-11-25')
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
