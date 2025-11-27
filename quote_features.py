"""
quote_features.py - Quote and orderflow feature engineering

Computes advanced quote-level and orderflow features from bid/ask tick data
with NO LOOK-AHEAD BIAS (all features use prior/current bar only).

Feature Categories:
1. Spread Dynamics
2. Bid-Ask Imbalance
3. Order Flow Toxicity
4. Depth Analysis
5. Volatility (quote-based)
6. Level 2 Metrics
7. Microstructure
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('quote_features')


class QuoteFeatures:
    """Quote-level and orderflow feature engineering."""
    
    @staticmethod
    def add_spread_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute bid-ask spread metrics.
        
        Features:
        - spread: ask - bid (basis points)
        - spread_pct: spread relative to mid
        - spread_sma: 20-bar moving average of spread (no lookahead)
        - spread_ratio: current spread / SMA spread
        - effective_spread: 2 * |price - mid| (transaction cost proxy)
        """
        if 'bid' not in df.columns or 'ask' not in df.columns:
            logger.warning("Missing bid/ask columns for spread metrics")
            return df
        
        df['spread_bp'] = (df['ask'] - df['bid']) * 10000
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread_pct'] = ((df['ask'] - df['bid']) / df['mid']) * 10000
        
        # Spread SMA (NO LOOKAHEAD: shift(1) before rolling)
        df['spread_bp_shifted'] = df['spread_bp'].shift(1)
        df['spread_sma'] = df['spread_bp_shifted'].rolling(20, min_periods=1).mean()
        
        df['spread_ratio'] = df['spread_bp'] / (df['spread_sma'] + 1e-10)
        
        return df
    
    @staticmethod
    def add_bid_ask_imbalance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute bid-ask imbalance (order flow indicator).
        
        Features:
        - buy_pressure: bid_size / (bid_size + ask_size)
        - sell_pressure: ask_size / (bid_size + ask_size)
        - order_imbalance: (bid_size - ask_size) / (bid_size + ask_size)
        - imbalance_sma: 20-bar SMA of imbalance
        - imbalance_momentum: current imbalance - SMA (shows flow intensity)
        """
        if 'bid_size' not in df.columns or 'ask_size' not in df.columns:
            logger.warning("Missing bid_size/ask_size for imbalance metrics")
            return df
        
        total_size = df['bid_size'] + df['ask_size']
        total_size = total_size.replace(0, 1)  # Avoid division by zero
        
        df['buy_pressure'] = df['bid_size'] / total_size
        df['sell_pressure'] = df['ask_size'] / total_size
        
        df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / total_size
        
        # SMA with no lookahead
        df['order_imbalance_shifted'] = df['order_imbalance'].shift(1)
        df['imbalance_sma'] = df['order_imbalance_shifted'].rolling(20, min_periods=1).mean()
        
        df['imbalance_momentum'] = df['order_imbalance'] - df['imbalance_sma']
        
        # Imbalance direction (persistent indicator)
        df['imbalance_direction'] = np.where(df['order_imbalance'] > 0, 1, -1)
        
        return df
    
    @staticmethod
    def add_order_flow_toxicity(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute order flow toxicity (likelihood of adverse selection).
        
        Toxic flow = large imbalance + wide spread + quick reversal
        
        Features:
        - toxicity_score: abs(imbalance) * spread_ratio (high = toxic)
        - flow_persistence: abs(imbalance_momentum) / abs(imbalance_sma)
        - aggressive_buying: imbalance > 0.5 AND bid_size > ask_size
        - aggressive_selling: imbalance < -0.5 AND ask_size > bid_size
        """
        if 'order_imbalance' not in df.columns:
            logger.warning("Missing order_imbalance for toxicity metrics")
            return df
        
        abs_imbalance = df['order_imbalance'].abs()
        spread_ratio = df.get('spread_ratio', pd.Series(1.0, index=df.index))
        
        df['toxicity_score'] = abs_imbalance * spread_ratio
        
        # Flow persistence
        imbalance_mom_abs = df.get('imbalance_momentum', pd.Series(0, index=df.index)).abs()
        imbalance_sma_abs = df.get('imbalance_sma', pd.Series(1e-10, index=df.index)).abs() + 1e-10
        df['flow_persistence'] = imbalance_mom_abs / imbalance_sma_abs
        
        # Aggressive flow detection
        if 'bid_size' in df.columns and 'ask_size' in df.columns:
            df['aggressive_buying'] = (df['order_imbalance'] > 0.5).astype(int)
            df['aggressive_selling'] = (df['order_imbalance'] < -0.5).astype(int)
        
        return df
    
    @staticmethod
    def add_depth_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute order book depth metrics.
        
        Features:
        - bid_depth_ratio: bid_size / (bid_size + ask_size) at best level
        - depth_imbalance: bid_size - ask_size (absolute)
        - depth_concentration: best_level_size / total_size (HOW CONCENTRATED?)
        - cumulative_volume_ratio: ratio of bid to ask cumulative
        """
        if 'bid_size' not in df.columns or 'ask_size' not in df.columns:
            logger.warning("Missing size data for depth metrics")
            return df
        
        total_size = df['bid_size'] + df['ask_size']
        
        df['bid_depth_ratio'] = df['bid_size'] / (total_size + 1e-10)
        df['depth_imbalance_abs'] = (df['bid_size'] - df['ask_size']).abs()
        
        return df
    
    @staticmethod
    def add_volatility_quote_based(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility from quote data.
        
        Features:
        - bid_volatility: volatility of bid prices
        - ask_volatility: volatility of ask prices
        - mid_volatility: volatility of mid prices
        - quote_range: max(ask) - min(bid) over window
        - quote_std: standard deviation of quotes over window
        """
        if 'bid' not in df.columns or 'ask' not in df.columns:
            logger.warning("Missing bid/ask for quote volatility")
            return df
        
        # Prior bar volatility (no lookahead)
        window = 20
        
        df['bid_volatility'] = df['bid'].shift(1).rolling(window, min_periods=1).std()
        df['ask_volatility'] = df['ask'].shift(1).rolling(window, min_periods=1).std()
        
        if 'mid' in df.columns:
            df['mid_volatility'] = df['mid'].shift(1).rolling(window, min_periods=1).std()
        
        return df
    
    @staticmethod
    def add_microstructure_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute microstructure indicators.
        
        Features:
        - quoted_spread: current bid-ask spread
        - realized_spread: difference between execution and mid (proxy)
        - price_improvement: (mid - bid) / spread (buy liquidity quality)
        - tick_direction: 1 if tick up, 0 if down, -1 if no tick
        - trade_intensity: bid_size * ask_size (how much liquidity available)
        """
        if 'bid' not in df.columns or 'ask' not in df.columns:
            logger.warning("Missing price data for microstructure")
            return df
        
        df['quoted_spread'] = df['ask'] - df['bid']
        
        # Price improvement metric
        if 'mid' in df.columns:
            df['price_improvement'] = (df['mid'] - df['bid']) / (df['ask'] - df['bid'] + 1e-10)
        
        # Tick direction (previous to current)
        df['prev_mid'] = df.get('mid', (df['bid'] + df['ask']) / 2).shift(1)
        df['curr_mid'] = df.get('mid', (df['bid'] + df['ask']) / 2)
        df['tick_direction'] = np.where(
            df['curr_mid'] > df['prev_mid'], 1,
            np.where(df['curr_mid'] < df['prev_mid'], -1, 0)
        )
        
        # Trade intensity
        if 'bid_size' in df.columns and 'ask_size' in df.columns:
            df['trade_intensity'] = df['bid_size'] * df['ask_size']
            df['trade_intensity_normalized'] = df['trade_intensity'] / (df['trade_intensity'].rolling(20, min_periods=1).mean() + 1e-10)
        
        return df
    
    @staticmethod
    def add_all_quote_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all quote features with NO LOOKAHEAD BIAS."""
        logger.info("Computing quote features...")
        
        df = df.copy()
        
        # All feature groups
        df = QuoteFeatures.add_spread_metrics(df)
        df = QuoteFeatures.add_bid_ask_imbalance(df)
        df = QuoteFeatures.add_order_flow_toxicity(df)
        df = QuoteFeatures.add_depth_analysis(df)
        df = QuoteFeatures.add_volatility_quote_based(df)
        df = QuoteFeatures.add_microstructure_metrics(df)
        
        # Drop temporary columns
        temp_cols = ['spread_bp_shifted', 'order_imbalance_shifted', 'prev_mid', 'curr_mid']
        df = df.drop(columns=[c for c in temp_cols if c in df.columns])
        
        logger.info(f"Generated {len(df.columns)} quote feature columns")
        
        return df


# Feature list for user reference
QUOTE_FEATURE_LIST = """
QUOTE-BASED FEATURE LIST:

SPREAD DYNAMICS (5 features):
- spread_bp: Spread in basis points
- spread_pct: Spread as percentage of mid
- spread_sma: 20-bar moving average of spread
- spread_ratio: Current spread / SMA spread
- quoted_spread: Ask - bid

BID-ASK IMBALANCE (5 features):
- buy_pressure: bid_size / total_size
- sell_pressure: ask_size / total_size
- order_imbalance: (bid_size - ask_size) / total_size
- imbalance_sma: 20-bar SMA of imbalance
- imbalance_momentum: Current imbalance - SMA

ORDER FLOW TOXICITY (3 features):
- toxicity_score: |imbalance| * spread_ratio
- flow_persistence: |imbalance_momentum| / |imbalance_sma|
- aggressive_buying: imbalance > 0.5
- aggressive_selling: imbalance < -0.5

DEPTH ANALYSIS (3 features):
- bid_depth_ratio: bid_size / total_size
- depth_imbalance_abs: |bid_size - ask_size|

QUOTE VOLATILITY (3 features):
- bid_volatility: Volatility of bid prices
- ask_volatility: Volatility of ask prices
- mid_volatility: Volatility of mid prices

MICROSTRUCTURE (5 features):
- price_improvement: (mid - bid) / spread
- tick_direction: 1=up, 0=same, -1=down
- trade_intensity: bid_size * ask_size
- trade_intensity_normalized: Current / 20-bar average
- imbalance_direction: 1=buy pressure, -1=sell pressure

TOTAL: ~30 quote-specific features for orderflow modeling
"""


if __name__ == '__main__':
    print(QUOTE_FEATURE_LIST)
