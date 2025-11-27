"""
Advanced Quote Features Engineering
Adds 12 critical missing quote features for high-frequency trading edge:
1. Microprice (weighted mid-price)
2. Microprice delta (returns across timeframes)
3. Quote imbalance (normalized)
4. Top-of-book volatility
5. Spread volatility metrics
6. Order flow volatility
7. Quote-book pressure gradient
8. Imbalance ratio of volume
9. Short-term liquidity shocks
10. Quote momentum
11. Queue-position proxy
12. Quote-based ATR
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class AdvancedQuoteFeatures:
    """Advanced quote feature engineering for high-frequency alpha"""
    
    @staticmethod
    def add_microprice(df):
        """
        Microprice = weighted mid-price incorporating imbalance
        microprice = (ask_price*bid_size + bid_price*ask_size) / (bid_size + ask_size)
        
        Predicts next-tick direction better than mid-price
        """
        logger.info("Computing microprice...")
        
        # Handle cases where we might not have raw bid/ask (aggregated data)
        if 'bid_first' in df.columns and 'ask_first' in df.columns:
            bid_price = df['bid_first']
            ask_price = df['ask_first']
        elif 'bid' in df.columns and 'ask' in df.columns:
            bid_price = df['bid']
            ask_price = df['ask']
        else:
            logger.warning("No bid/ask columns found, using open/close")
            bid_price = df['open']
            ask_price = df['close']
            df['microprice'] = (bid_price + ask_price) / 2
            return df
        
        # Get sizes
        bid_size = df.get('bid_size_sum', df.get('bid_size_mean', 1.0))
        ask_size = df.get('ask_size_sum', df.get('ask_size_mean', 1.0))
        
        # Ensure no division by zero
        total_size = bid_size + ask_size
        total_size = total_size.replace(0, 1)
        
        df['microprice'] = (ask_price * bid_size + bid_price * ask_size) / total_size
        logger.info("✓ Microprice added")
        
        return df
    
    @staticmethod
    def add_microprice_delta(df):
        """
        Microprice returns at different periods
        Captures momentum at multiple scales
        """
        logger.info("Computing microprice delta...")
        
        if 'microprice' not in df.columns:
            logger.warning("Microprice not found, skipping delta")
            return df
        
        # 1-bar return
        df['microprice_return_1'] = df['microprice'].pct_change() * 10000  # in bps
        
        # 5-bar return
        df['microprice_return_5'] = (df['microprice'].shift(0) - df['microprice'].shift(5)) / df['microprice'].shift(5) * 10000
        
        # 15-bar return
        df['microprice_return_15'] = (df['microprice'].shift(0) - df['microprice'].shift(15)) / df['microprice'].shift(15) * 10000
        
        logger.info("✓ Microprice delta added")
        return df
    
    @staticmethod
    def add_quote_imbalance(df):
        """
        Normalized quote imbalance
        qi = (bid_size - ask_size) / (bid_size + ask_size)
        
        Strongest predictor in HFT
        """
        logger.info("Computing quote imbalance...")
        
        if 'bid_size_sum' not in df.columns or 'ask_size_sum' not in df.columns:
            logger.warning("bid/ask_size columns not found, computing from pressure")
            if 'buy_pressure' in df.columns and 'sell_pressure' in df.columns:
                df['quote_imbalance'] = df['buy_pressure'] - df['sell_pressure']
            else:
                logger.warning("Cannot compute quote imbalance")
                return df
        else:
            bid_size = df['bid_size_sum']
            ask_size = df['ask_size_sum']
            total = bid_size + ask_size
            total = total.replace(0, 1)
            df['quote_imbalance'] = (bid_size - ask_size) / total
        
        logger.info("✓ Quote imbalance added")
        return df
    
    @staticmethod
    def add_top_of_book_volatility(df):
        """
        Returns of bid/ask quotes
        bid_return, ask_return, mid_return
        """
        logger.info("Computing top-of-book volatility...")
        
        if 'bid_first' in df.columns:
            df['bid_return'] = df['bid_first'].pct_change() * 10000
        if 'ask_first' in df.columns:
            df['ask_return'] = df['ask_first'].pct_change() * 10000
        if 'mid_price_mean' in df.columns:
            df['mid_return'] = df['mid_price_mean'].pct_change() * 10000
        
        # Rolling volatility of returns
        if 'bid_return' in df.columns:
            df['bid_vol'] = df['bid_return'].rolling(20, min_periods=5).std()
        if 'ask_return' in df.columns:
            df['ask_vol'] = df['ask_return'].rolling(20, min_periods=5).std()
        if 'mid_return' in df.columns:
            df['mid_vol'] = df['mid_return'].rolling(20, min_periods=5).std()
        
        logger.info("✓ Top-of-book volatility added")
        return df
    
    @staticmethod
    def add_spread_volatility(df):
        """
        Spread metrics: zscore, volatility, skew
        Spread spikes often precede direction
        """
        logger.info("Computing spread volatility...")
        
        if 'spread' not in df.columns:
            logger.warning("Spread not found, computing from bid/ask")
            if 'ask_first' in df.columns and 'bid_first' in df.columns:
                df['spread'] = df['ask_first'] - df['bid_first']
            else:
                logger.warning("Cannot compute spread")
                return df
        
        # Spread zscore
        spread_mean = df['spread'].rolling(50, min_periods=10).mean()
        spread_std = df['spread'].rolling(50, min_periods=10).std()
        spread_std = spread_std.replace(0, 1)
        df['spread_zscore'] = (df['spread'] - spread_mean) / spread_std
        
        # Spread volatility
        df['spread_volatility'] = df['spread'].rolling(20, min_periods=5).std()
        
        # Spread skew (3rd moment)
        df['spread_skew'] = df['spread'].rolling(20, min_periods=5).skew()
        
        logger.info("✓ Spread volatility metrics added")
        return df
    
    @staticmethod
    def add_orderflow_volatility(df):
        """
        Order flow volatility = std(buy_pressure - sell_pressure)
        Spikes indicate aggressive entry
        """
        logger.info("Computing order flow volatility...")
        
        if 'buy_pressure' not in df.columns or 'sell_pressure' not in df.columns:
            logger.warning("buy/sell_pressure not found")
            return df
        
        orderflow = df['buy_pressure'] - df['sell_pressure']
        df['orderflow_volatility'] = orderflow.rolling(20, min_periods=5).std()
        
        logger.info("✓ Order flow volatility added")
        return df
    
    @staticmethod
    def add_pressure_gradient(df):
        """
        Quote-book pressure gradient
        pressure_gradient = (buy_pressure - sell_pressure) / spread
        
        Missing and major edge
        """
        logger.info("Computing pressure gradient...")
        
        if 'buy_pressure' not in df.columns or 'sell_pressure' not in df.columns or 'spread' not in df.columns:
            logger.warning("Required columns not found for pressure gradient")
            return df
        
        numerator = df['buy_pressure'] - df['sell_pressure']
        denominator = df['spread']
        denominator = denominator.replace(0, 0.00001)  # Avoid division by zero
        
        df['pressure_gradient'] = numerator / denominator
        
        logger.info("✓ Pressure gradient added")
        return df
    
    @staticmethod
    def add_volume_imbalance(df):
        """
        Imbalance ratio of volume
        volume_imbalance = bid_size_sum / (bid_size_sum + ask_size_sum)
        """
        logger.info("Computing volume imbalance...")
        
        if 'bid_size_sum' not in df.columns or 'ask_size_sum' not in df.columns:
            logger.warning("bid/ask_size_sum not found")
            return df
        
        total_volume = df['bid_size_sum'] + df['ask_size_sum']
        total_volume = total_volume.replace(0, 1)
        
        df['volume_imbalance'] = df['bid_size_sum'] / total_volume
        
        logger.info("✓ Volume imbalance added")
        return df
    
    @staticmethod
    def add_liquidity_shocks(df):
        """
        Short-term liquidity shocks
        liquidity_drop = bid_size_mean_rolling5 - bid_size_mean_rolling1
        
        Predicts stop hunts on XAUUSD
        """
        logger.info("Computing liquidity shocks...")
        
        if 'bid_size_mean' in df.columns:
            bid_rolling_5 = df['bid_size_mean'].rolling(5, min_periods=2).mean()
            bid_rolling_1 = df['bid_size_mean'].rolling(1, min_periods=1).mean()
            df['bid_liquidity_shock'] = bid_rolling_5 - bid_rolling_1
        
        if 'ask_size_mean' in df.columns:
            ask_rolling_5 = df['ask_size_mean'].rolling(5, min_periods=2).mean()
            ask_rolling_1 = df['ask_size_mean'].rolling(1, min_periods=1).mean()
            df['ask_liquidity_shock'] = ask_rolling_5 - ask_rolling_1
        
        logger.info("✓ Liquidity shocks added")
        return df
    
    @staticmethod
    def add_quote_momentum(df):
        """
        Quote momentum: rolling slope of microprice/midprice
        Captures directional bias in order flow
        """
        logger.info("Computing quote momentum...")
        
        # Microprice momentum
        if 'microprice' in df.columns:
            window = 10
            microprice_vals = df['microprice'].values
            microprice_slopes = []
            
            for i in range(len(microprice_vals)):
                if i < window:
                    x = np.arange(i + 1)
                    y = microprice_vals[:i + 1]
                else:
                    x = np.arange(window)
                    y = microprice_vals[i - window + 1:i + 1]
                
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    microprice_slopes.append(slope)
                else:
                    microprice_slopes.append(0)
            
            df['microprice_momentum'] = microprice_slopes
        
        # Midprice momentum
        if 'mid_price_mean' in df.columns:
            window = 10
            midprice_vals = df['mid_price_mean'].values
            midprice_slopes = []
            
            for i in range(len(midprice_vals)):
                if i < window:
                    x = np.arange(i + 1)
                    y = midprice_vals[:i + 1]
                else:
                    x = np.arange(window)
                    y = midprice_vals[i - window + 1:i + 1]
                
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    midprice_slopes.append(slope)
                else:
                    midprice_slopes.append(0)
            
            df['midprice_momentum'] = midprice_slopes
        
        logger.info("✓ Quote momentum added")
        return df
    
    @staticmethod
    def add_queue_position_proxy(df):
        """
        Queue position proxy using bid/ask shifts
        bid_queue_shift = bid_last - bid_first
        ask_queue_shift = ask_last - ask_first
        
        Extremely predictive
        """
        logger.info("Computing queue position proxy...")
        
        if 'bid_first' in df.columns and 'bid_last' in df.columns:
            df['bid_queue_shift'] = df['bid_last'] - df['bid_first']
        
        if 'ask_first' in df.columns and 'ask_last' in df.columns:
            df['ask_queue_shift'] = df['ask_last'] - df['ask_first']
        
        # Also compute size shifts
        if 'bid_size_first' in df.columns and 'bid_size_last' in df.columns:
            df['bid_size_shift'] = df['bid_size_last'] - df['bid_size_first']
        
        if 'ask_size_first' in df.columns and 'ask_size_last' in df.columns:
            df['ask_size_shift'] = df['ask_size_last'] - df['ask_size_first']
        
        logger.info("✓ Queue position proxy added")
        return df
    
    @staticmethod
    def add_quote_based_atr(df):
        """
        Quote-based ATR using mid-price
        More accurate than candle ATR for 1T-5T
        
        quote_ATR = rolling(mean(abs(mid_price - mid_price.shift(1))))
        """
        logger.info("Computing quote-based ATR...")
        
        if 'mid_price_mean' not in df.columns:
            logger.warning("mid_price_mean not found")
            return df
        
        # Absolute changes
        price_changes = df['mid_price_mean'].diff().abs()
        
        # Rolling average (different windows for different purposes)
        df['quote_atr_5'] = price_changes.rolling(5, min_periods=1).mean()
        df['quote_atr_10'] = price_changes.rolling(10, min_periods=1).mean()
        df['quote_atr_20'] = price_changes.rolling(20, min_periods=1).mean()
        
        # Also compute high-low range as alternative
        if 'high' in df.columns and 'low' in df.columns:
            df['quote_hl_range'] = df['high'] - df['low']
        
        logger.info("✓ Quote-based ATR added")
        return df
    
    @staticmethod
    def add_all_advanced_features(df):
        """Apply all advanced features"""
        logger.info("\n" + "="*70)
        logger.info("ADDING ADVANCED QUOTE FEATURES (12 critical indicators)")
        logger.info("="*70 + "\n")
        
        df = AdvancedQuoteFeatures.add_microprice(df)
        df = AdvancedQuoteFeatures.add_microprice_delta(df)
        df = AdvancedQuoteFeatures.add_quote_imbalance(df)
        df = AdvancedQuoteFeatures.add_top_of_book_volatility(df)
        df = AdvancedQuoteFeatures.add_spread_volatility(df)
        df = AdvancedQuoteFeatures.add_orderflow_volatility(df)
        df = AdvancedQuoteFeatures.add_pressure_gradient(df)
        df = AdvancedQuoteFeatures.add_volume_imbalance(df)
        df = AdvancedQuoteFeatures.add_liquidity_shocks(df)
        df = AdvancedQuoteFeatures.add_quote_momentum(df)
        df = AdvancedQuoteFeatures.add_queue_position_proxy(df)
        df = AdvancedQuoteFeatures.add_quote_based_atr(df)
        
        # Forward-fill NaNs from rolling windows
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"\n✓ All advanced features added")
        logger.info(f"✓ Total columns: {len(df.columns)}")
        
        return df


def process_quote_files():
    """Process all quote parquet files with advanced features"""
    data_dir = Path('feature_store/C:XAU-USD/quotes')
    
    timeframes = ['1T', '5T', '15T', '30T']
    
    logger.info("\n" + "="*70)
    logger.info("PROCESSING QUOTE FILES WITH ADVANCED FEATURES")
    logger.info("="*70)
    
    for tf in timeframes:
        input_file = data_dir / f'C:XAU-USD_{tf}_quotes.parquet'
        output_file = data_dir / f'C:XAU-USD_{tf}_quotes_advanced.parquet'
        
        if not input_file.exists():
            logger.warning(f"File not found: {input_file}")
            continue
        
        logger.info(f"\nProcessing {tf}...")
        df = pd.read_parquet(input_file)
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Add advanced features
        df = AdvancedQuoteFeatures.add_all_advanced_features(df)
        
        # Save
        df.to_parquet(output_file, compression='snappy', index=False)
        logger.info(f"✓ Saved: {output_file}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ ADVANCED FEATURES PROCESSING COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    process_quote_files()
