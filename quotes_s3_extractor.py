"""
quotes_s3_extractor.py - Extract quote-level data from Polygon S3

Downloads quotes_v1 daily files from S3 and aggregates quote data
to OHLC timeframes for use in orderflow strategy development.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta
import boto3
from botocore.config import Config
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('quotes_extractor')


class PolygonS3QuoteExtractor:
    """Extract quote tick data from Polygon S3."""
    
    def __init__(self, symbol: str = "XAUUSD"):
        """
        symbol: e.g. "XAUUSD" (NOT "C:XAU-USD")
        """
        self.symbol = symbol
        self.bucket = "polygon-quotes-v1"
        
        # Configure S3 with retries and timeout
        s3_config = Config(
            retries={'max_attempts': 3},
            connect_timeout=10,
            read_timeout=60,
        )
        self.s3 = boto3.client('s3', config=s3_config)
    
    def download_quote_files(self, date_range: tuple) -> list:
        """
        Download quote files for date range from S3.
        
        date_range: (start_date, end_date) as datetime or "2020-01-01" strings
        Returns: list of dataframes with quote data
        """
        if isinstance(date_range[0], str):
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
        else:
            start_date = date_range[0]
            end_date = date_range[1]
        
        logger.info(f"Fetching {self.symbol} quotes from {start_date.date()} to {end_date.date()}")
        
        all_quotes = []
        current_date = start_date
        file_count = 0
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y/%m/%d")
            file_key = f"{date_str}/quotes_v1_forex_{self.symbol}.csv.gz"
            
            try:
                logger.debug(f"Attempting to download: {file_key}")
                
                response = self.s3.get_object(Bucket=self.bucket, Key=file_key)
                df = pd.read_csv(response['Body'], compression='gzip')
                
                # Parse timestamp columns
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
                df['quote_at'] = pd.to_datetime(df['quote_at'], unit='ns')
                
                logger.info(f"  ✓ {current_date.date()}: {len(df)} quotes")
                all_quotes.append(df)
                file_count += 1
                
            except self.s3.exceptions.NoSuchKey:
                logger.debug(f"  ✗ File not found: {file_key}")
            except Exception as e:
                logger.warning(f"  ✗ Error downloading {file_key}: {e}")
            
            current_date += timedelta(days=1)
        
        if not all_quotes:
            logger.warning("No quote files downloaded!")
            return []
        
        logger.info(f"Downloaded {file_count} daily quote files")
        
        # Concatenate all
        df_all = pd.concat(all_quotes, ignore_index=True)
        df_all = df_all.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Total quotes: {len(df_all)}")
        logger.info(f"Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
        
        return df_all
    
    @staticmethod
    def aggregate_to_ohlc(quotes_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Aggregate quote-level data to OHLC bars + quote features.
        
        timeframe: "1T", "5T", "15T", "30T"
        """
        # Set timestamp as index
        df = quotes_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample OHLC
        ohlc = df['mid_price'].resample(timeframe).ohlc()
        
        # Additional quote features per timeframe
        agg_dict = {
            'bid': ['first', 'last', 'min', 'max'],
            'ask': ['first', 'last', 'min', 'max'],
            'mid_price': ['mean', 'std'],
            'bid_size': ['sum', 'mean', 'max'],
            'ask_size': ['sum', 'mean', 'max'],
        }
        
        # Only aggregate columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        quote_features = df.resample(timeframe).agg(agg_dict)
        
        # Flatten multi-level columns
        quote_features.columns = ['_'.join(col).strip('_') for col in quote_features.columns.values]
        
        # Combine
        result = ohlc.copy()
        result = pd.concat([result, quote_features], axis=1)
        result = result.reset_index()
        
        # Compute bid-ask metrics
        result['spread'] = result['ask_first'] - result['bid_first'] if 'ask_first' in result.columns else np.nan
        result['spread_pct'] = (result['spread'] / result['mid_price_mean']) * 10000 if 'mid_price_mean' in result.columns else np.nan
        
        # Bid-ask imbalance
        if 'bid_size_sum' in result.columns and 'ask_size_sum' in result.columns:
            total_size = result['bid_size_sum'] + result['ask_size_sum']
            result['buy_pressure'] = result['bid_size_sum'] / total_size
            result['sell_pressure'] = result['ask_size_sum'] / total_size
        
        logger.info(f"Aggregated {timeframe}: {len(result)} bars, {len(result.columns)} features")
        
        return result
    
    @staticmethod
    def save_quote_parquets(quotes_df: pd.DataFrame, symbol: str, 
                           output_dir: str = "feature_store"):
        """Save quote data as timeframe-specific parquets."""
        out_path = Path(output_dir) / symbol / "quotes"
        out_path.mkdir(parents=True, exist_ok=True)
        
        timeframes = ['1T', '5T', '15T', '30T']
        
        for tf in timeframes:
            df_tf = PolygonS3QuoteExtractor.aggregate_to_ohlc(quotes_df, tf)
            
            save_file = out_path / f"{symbol}_{tf}_quotes.parquet"
            df_tf.to_parquet(save_file)
            
            logger.info(f"✓ Saved {save_file}: {len(df_tf)} bars")


def main():
    import argparse
    
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='XAUUSD', help='Symbol in S3 format (e.g. XAUUSD)')
    p.add_argument('--start', default='2020-01-01')
    p.add_argument('--end', default='2025-11-30')
    p.add_argument('--output', default='feature_store')
    args = p.parse_args()
    
    extractor = PolygonS3QuoteExtractor(args.symbol)
    
    try:
        # Download quotes
        quotes_df = extractor.download_quote_files((args.start, args.end))
        
        if quotes_df is not None and len(quotes_df) > 0:
            # Save parquets
            extractor.save_quote_parquets(quotes_df, f"C:{args.symbol[-3:].upper()}-{args.symbol[:3].upper()}", args.output)
            logger.info("\n✓ Quote extraction complete")
        else:
            logger.warning("No data downloaded")
    
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
