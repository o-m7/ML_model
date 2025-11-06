#!/usr/bin/env python3
"""
Continuous Feature Calculator and Uploader to Supabase
=======================================================

Fetches live data from Polygon API, calculates all features,
and uploads them to Supabase for real-time signal generation.

Usage:
    python upload_features_to_supabase.py --symbol XAUUSD --timeframe 1H
    python upload_features_to_supabase.py --symbols XAUUSD,EURUSD --timeframes 1H,4H --continuous
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import logging

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

# Import feature builders
from intraday_system.live import get_live_data
from intraday_system.features.builders import FeatureBuilder
from intraday_system.features.regime import RegimeFeatures
from add_talib_features import TALibEnricher

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_uploader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FeatureUploader:
    """Fetch, calculate, and upload features to Supabase."""
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        polygon_api_key: Optional[str] = None
    ):
        """Initialize the uploader."""
        # Supabase client
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_SERVICE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Polygon API key
        self.polygon_api_key = polygon_api_key or os.getenv('POLYGON_API_KEY')
        if not self.polygon_api_key:
            raise ValueError("POLYGON_API_KEY must be set in .env")
        
        logger.info("✓ FeatureUploader initialized")
    
    def fetch_and_calculate_features(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int = 200
    ) -> Optional[Dict]:
        """
        Fetch live data and calculate all features.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Timeframe (e.g., '1H')
            n_bars: Number of bars to fetch
            
        Returns:
            Dictionary with OHLCV and calculated features
        """
        try:
            # Fetch live data
            logger.info(f"Fetching {n_bars} bars for {symbol} {timeframe}...")
            df = get_live_data(
                symbol=symbol,
                timeframe=timeframe,
                n_bars=n_bars,
                api_key=self.polygon_api_key
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return None
            
            # Ensure proper data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(np.float64)
            
            # Calculate features
            logger.info(f"Calculating features for {symbol} {timeframe}...")
            
            # Base features
            builder = FeatureBuilder()
            df = builder.build_all(df)
            
            # TA-Lib features (suppress warnings)
            import warnings
            warnings.filterwarnings('ignore')
            df = TALibEnricher.add_all_indicators(df)
            warnings.filterwarnings('default')
            
            # Regime features
            regime = RegimeFeatures()
            df = regime.add_all(df)
            
            # Get latest bar
            latest_row = df.iloc[-1]
            
            # Extract OHLCV
            ohlcv = {
                'timestamp': str(latest_row['timestamp']),
                'open': float(latest_row['open']),
                'high': float(latest_row['high']),
                'low': float(latest_row['low']),
                'close': float(latest_row['close']),
                'volume': float(latest_row['volume']),
            }
            
            # Extract all features (exclude OHLCV and timestamp)
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            features = {}
            
            for col in df.columns:
                if col not in exclude_cols:
                    val = latest_row[col]
                    # Convert to Python native types for JSON serialization
                    if pd.isna(val):
                        features[col] = None
                    elif isinstance(val, (np.integer, np.int64)):
                        features[col] = int(val)
                    elif isinstance(val, (np.floating, np.float64)):
                        features[col] = float(val)
                    elif isinstance(val, bool):
                        features[col] = bool(val)
                    else:
                        features[col] = val
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                **ohlcv,
                'features': features,
                'feature_count': len(features)
            }
            
            logger.info(f"✓ Calculated {len(features)} features for {symbol} {timeframe}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Error calculating features for {symbol} {timeframe}: {e}")
            return None
    
    def upload_to_supabase(self, data: Dict) -> bool:
        """
        Upload feature data to Supabase.
        
        Args:
            data: Dictionary with symbol, timeframe, OHLCV, and features
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare record
            record = {
                'symbol': data['symbol'],
                'timeframe': data['timeframe'],
                'timestamp': data['timestamp'],
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume'],
                'features': json.dumps(data['features']),  # JSONB
                'feature_count': data['feature_count'],
                'data_source': 'Polygon API'
            }
            
            # Upsert (insert or update if exists)
            response = self.supabase.table('features').upsert(
                record,
                on_conflict='symbol,timeframe,timestamp'
            ).execute()
            
            logger.info(f"✓ Uploaded features for {data['symbol']} {data['timeframe']} at {data['timestamp']}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Upload failed for {data['symbol']} {data['timeframe']}: {e}")
            return False
    
    def process_symbol_timeframe(self, symbol: str, timeframe: str) -> bool:
        """Process a single symbol/timeframe combination."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {symbol} {timeframe}")
        logger.info(f"{'='*70}")
        
        # Fetch and calculate
        data = self.fetch_and_calculate_features(symbol, timeframe)
        if not data:
            return False
        
        # Upload
        success = self.upload_to_supabase(data)
        return success
    
    def run_continuous(
        self,
        symbols: List[str],
        timeframes: List[str],
        interval_seconds: int = 900  # 15 minutes
    ):
        """
        Run continuous feature updates.
        
        Args:
            symbols: List of symbols to process
            timeframes: List of timeframes to process
            interval_seconds: Seconds to wait between updates
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Starting continuous feature uploader")
        logger.info(f"{'='*70}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Update interval: {interval_seconds} seconds ({interval_seconds/60:.1f} minutes)")
        logger.info(f"{'='*70}\n")
        
        iteration = 0
        
        while True:
            iteration += 1
            start_time = time.time()
            
            logger.info(f"\n{'#'*70}")
            logger.info(f"ITERATION {iteration} - {datetime.now(timezone.utc).isoformat()}")
            logger.info(f"{'#'*70}\n")
            
            success_count = 0
            total_count = 0
            
            # Process each symbol/timeframe combination
            for symbol in symbols:
                for timeframe in timeframes:
                    total_count += 1
                    try:
                        if self.process_symbol_timeframe(symbol, timeframe):
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing {symbol} {timeframe}: {e}")
            
            elapsed = time.time() - start_time
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Iteration {iteration} complete: {success_count}/{total_count} successful")
            logger.info(f"Elapsed time: {elapsed:.1f}s")
            logger.info(f"Next update in {interval_seconds}s...")
            logger.info(f"{'='*70}\n")
            
            # Wait for next interval
            time.sleep(interval_seconds)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Upload calculated features to Supabase continuously'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        help='Single symbol to process (e.g., XAUUSD)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated symbols (e.g., XAUUSD,EURUSD,GBPUSD)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        help='Single timeframe to process (e.g., 1H)'
    )
    parser.add_argument(
        '--timeframes',
        type=str,
        help='Comma-separated timeframes (e.g., 1H,4H)'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuously (default: single run)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=900,
        help='Update interval in seconds (default: 900 = 15 minutes)'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        # Default symbols
        symbols = ['XAUUSD', 'EURUSD', 'GBPUSD']
        logger.info(f"No symbols specified, using defaults: {symbols}")
    
    # Parse timeframes
    if args.timeframes:
        timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    elif args.timeframe:
        timeframes = [args.timeframe]
    else:
        # Default timeframes
        timeframes = ['1H', '4H']
        logger.info(f"No timeframes specified, using defaults: {timeframes}")
    
    # Initialize uploader
    try:
        uploader = FeatureUploader()
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        logger.error("Make sure SUPABASE_URL, SUPABASE_SERVICE_KEY, and POLYGON_API_KEY are set in .env")
        return 1
    
    # Run
    if args.continuous:
        # Continuous mode
        uploader.run_continuous(symbols, timeframes, args.interval)
    else:
        # Single run mode
        success_count = 0
        total_count = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                total_count += 1
                if uploader.process_symbol_timeframe(symbol, timeframe):
                    success_count += 1
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ Complete: {success_count}/{total_count} successful")
        logger.info(f"{'='*70}\n")
        
        return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())

