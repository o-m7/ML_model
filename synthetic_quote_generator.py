"""
synthetic_quote_generator.py - Generate synthetic quote data from OHLCV

Since quote-level data is not available, we synthesize realistic bid/ask 
from the OHLCV candles we already have. This maintains full data integrity
while providing quote-like features for testing the orderflow pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('quote_synthesizer')


class QuoteSynthesizer:
    """Generate synthetic quote-level data from OHLCV candles."""
    
    @staticmethod
    def synthesize_quotes_from_ohlcv(df_ohlcv: pd.DataFrame, quotes_per_candle: int = 20) -> pd.DataFrame:
        """
        Create realistic bid/ask quotes from OHLCV data.
        
        For each OHLCV candle, generate multiple quotes that:
        1. Follow the High/Low price range
        2. Have realistic bid-ask spreads
        3. Simulate order book depth
        4. Maintain OHLCV properties
        """
        logger.info(f"Synthesizing {quotes_per_candle} quotes per candle...")
        
        df = df_ohlcv.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        all_quotes = []
        
        for idx, row in df.iterrows():
            candle_start = row['timestamp']
            # Determine candle timeframe from data
            if idx < len(df) - 1:
                next_start = df.iloc[idx + 1]['timestamp']
                candle_duration = next_start - candle_start
            else:
                # Estimate from previous candles
                candle_duration = pd.Timedelta(minutes=5)  # Default to 5T
            
            candle_end = candle_start + candle_duration
            
            # Generate times for quotes within this candle
            quote_times = pd.date_range(start=candle_start, end=candle_end, periods=quotes_per_candle, inclusive='left')
            
            high = row['high']
            low = row['low']
            open_ = row['open']
            close = row['close']
            volume = row.get('volume', 1.0)
            
            # Base spread (0.1-0.3% depending on volatility)
            candle_volatility = (high - low) / low if low > 0 else 0.001
            spread_pct = np.clip(0.001 + candle_volatility * 0.1, 0.0001, 0.005)
            
            for i, quote_time in enumerate(quote_times):
                # Interpolate price movement through candle
                candle_progress = i / quotes_per_candle
                
                # Price path: open → high/low → close
                if candle_progress < 0.5:
                    # First half: move toward high or low
                    target = high if close > open_ else low
                    mid_price = open_ + (target - open_) * (candle_progress * 2)
                else:
                    # Second half: move toward close
                    current_target = high if close > open_ else low
                    mid_price = current_target + (close - current_target) * ((candle_progress - 0.5) * 2)
                
                mid_price = np.clip(mid_price, low, high)
                
                # Bid-ask around mid
                spread = mid_price * spread_pct
                bid = mid_price - spread / 2
                ask = mid_price + spread / 2
                
                # Order book depth (simulated sizes)
                bid_size = np.random.lognormal(mean=np.log(volume / quotes_per_candle), sigma=0.5)
                ask_size = np.random.lognormal(mean=np.log(volume / quotes_per_candle), sigma=0.5)
                
                quote = {
                    'timestamp': quote_time,
                    'quote_at': quote_time,
                    'bid': bid,
                    'ask': ask,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'mid_price': mid_price,
                }
                
                all_quotes.append(quote)
        
        df_quotes = pd.DataFrame(all_quotes)
        logger.info(f"Generated {len(df_quotes):,} quotes from {len(df):,} candles")
        logger.info(f"Date range: {df_quotes['timestamp'].min()} to {df_quotes['timestamp'].max()}")
        
        return df_quotes
    
    @staticmethod
    def aggregate_quotes_to_ohlc(quotes_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate synthetic quotes back to OHLC bars."""
        df = quotes_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df_idx = df.set_index('timestamp')
        
        # OHLC from mid prices
        ohlc = df_idx['mid_price'].resample(timeframe).ohlc()
        
        # Volume
        volume = df_idx.resample(timeframe).size()
        
        # Quote-level aggregations
        agg_funcs = {
            'bid': ['first', 'last', 'min', 'max'],
            'ask': ['first', 'last', 'min', 'max'],
            'mid_price': ['mean', 'std'],
            'bid_size': ['sum', 'mean', 'max'],
            'ask_size': ['sum', 'mean', 'max'],
        }
        
        quote_agg = df_idx.resample(timeframe).agg(agg_funcs)
        quote_agg.columns = ['_'.join(col).strip() for col in quote_agg.columns]
        
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
            total = result['bid_size_sum'] + result['ask_size_sum']
            total = total.replace(0, 1)
            result['buy_pressure'] = result['bid_size_sum'] / total
            result['sell_pressure'] = result['ask_size_sum'] / total
        
        logger.info(f"Aggregated to {timeframe}: {len(result)} bars, {len(result.columns)} features")
        return result


def main():
    import argparse
    
    p = argparse.ArgumentParser(description='Generate synthetic quote data from OHLCV')
    p.add_argument('--input-dir', default='feature_store/C:XAU-USD', help='Input OHLCV directory')
    p.add_argument('--output-dir', default='feature_store/C:XAU-USD/quotes', help='Output quotes directory')
    p.add_argument('--symbol', default='C:XAU-USD', help='Symbol name')
    args = p.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("SYNTHETIC QUOTE GENERATOR")
    logger.info("=" * 70)
    
    # Load OHLCV data
    ohlcv_file = input_path / f"{args.symbol}_5T.parquet"
    
    if not ohlcv_file.exists():
        logger.error(f"OHLCV file not found: {ohlcv_file}")
        return
    
    logger.info(f"\nLoading OHLCV data: {ohlcv_file}")
    df_ohlcv = pd.read_parquet(ohlcv_file)
    df_ohlcv = df_ohlcv.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded: {len(df_ohlcv)} candles")
    logger.info(f"Date range: {df_ohlcv['timestamp'].min()} to {df_ohlcv['timestamp'].max()}")
    
    # Generate synthetic quotes
    logger.info("\n[SYNTHESIS]")
    df_quotes = QuoteSynthesizer.synthesize_quotes_from_ohlcv(df_ohlcv, quotes_per_candle=20)
    
    # Save raw quotes
    quotes_raw_file = output_path / f"{args.symbol}_quotes_raw.parquet"
    df_quotes.to_parquet(quotes_raw_file, compression='snappy', index=False)
    logger.info(f"\n✓ Saved raw quotes: {quotes_raw_file}")
    
    # Aggregate to timeframes
    logger.info("\n[AGGREGATION]")
    for tf in ['1T', '5T', '15T', '30T']:
        logger.info(f"\nAggregating to {tf}...")
        df_tf = QuoteSynthesizer.aggregate_quotes_to_ohlc(df_quotes, tf)
        
        tf_file = output_path / f"{args.symbol}_{tf}_quotes.parquet"
        df_tf.to_parquet(tf_file, compression='snappy', index=False)
        logger.info(f"✓ Saved: {tf_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ SYNTHETIC QUOTE GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total quotes: {len(df_quotes):,}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Date range: {df_quotes['timestamp'].min()} to {df_quotes['timestamp'].max()}")


if __name__ == '__main__':
    main()
