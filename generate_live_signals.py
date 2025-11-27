#!/usr/bin/env python3
"""
Live Signal Generator - Production Ready
Uses all 20 models trained on Nov 26 data (1T, 5T, 15T, 30T)
"""
import sys
import os
from pathlib import Path
import pickle
import logging
import time
import asyncio
from datetime import datetime

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from polygon_connector import PolygonRESTClient
from dotenv import load_dotenv

load_dotenv()


class LiveSignalGenerator:
    """Generate live trading signals using trained ML models"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = {}
        self.scalers = {}
        self.features = {}
        
    def load_models(self):
        """Load all trained models, scalers, and feature lists"""
        logger.info("Loading trained models...")
        
        timeframes = ['1T', '5T', '15T', '30T']
        
        # Load OHLCV models
        for tf in timeframes:
            model_path = f'artifacts/ohlcv_model_{tf}_xgb.pkl'
            scaler_path = f'artifacts/ohlcv_scaler_{tf}.pkl'
            features_path = f'artifacts/ohlcv_features_{tf}.txt'
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[f'ohlcv_{tf}'] = pickle.load(f)
                logger.info(f"  ✓ Loaded OHLCV model ({tf})")
                
                # Try to load scaler (optional)
                if os.path.exists(scaler_path):
                    try:
                        with open(scaler_path, 'rb') as f:
                            self.scalers[f'ohlcv_{tf}'] = pickle.load(f)
                    except Exception as e:
                        logger.warning(f"    Could not load scaler for {tf}: {e}")
                
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.features[f'ohlcv_{tf}'] = [line.strip() for line in f]
        
        # Load Quote models
        for tf in timeframes:
            model_path = f'artifacts/quote_model_{tf}_xgb.pkl'
            scaler_path = f'artifacts/quote_scaler_{tf}.pkl'
            features_path = f'artifacts/quote_features_{tf}.txt'
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[f'quote_{tf}'] = pickle.load(f)
                logger.info(f"  ✓ Loaded Quote model ({tf})")
                
                # Try to load scaler (optional)
                if os.path.exists(scaler_path):
                    try:
                        with open(scaler_path, 'rb') as f:
                            self.scalers[f'quote_{tf}'] = pickle.load(f)
                    except Exception as e:
                        logger.warning(f"    Could not load scaler for {tf}: {e}")
                
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.features[f'quote_{tf}'] = [line.strip() for line in f]
        
        logger.info(f"  ✓ Total models loaded: {len(self.models)}")
        
    async def fetch_live_data(self):
        """Fetch latest quotes and bars from Polygon"""
        async with PolygonRESTClient(self.api_key) as client:
            # Fetch quote
            quote = await client.get_latest_quote("C:XAU-USD")
            
            # Fetch bars for each timeframe
            bars = {}
            for tf_name, multiplier in [('1T', 1), ('5T', 5), ('15T', 15), ('30T', 30)]:
                bars_df = await client.get_historical_bars(
                    "C:XAUUSD",
                    timespan="minute",
                    multiplier=multiplier,
                    limit=100  # Get enough history for features
                )
                if bars_df is not None and len(bars_df) > 0:
                    bars[tf_name] = bars_df
            
            return quote, bars
    
    def compute_features_from_bars(self, bars_df: pd.DataFrame, timeframe: str) -> dict:
        """Compute features from OHLCV bars"""
        if len(bars_df) < 50:
            return None
        
        # Use last row for current features
        latest = bars_df.iloc[-1]
        
        # Basic OHLCV features
        features = {
            'close': latest['close'],
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'volume': latest['volume'],
        }
        
        # Returns
        if len(bars_df) >= 2:
            features['returns'] = (latest['close'] - bars_df.iloc[-2]['close']) / bars_df.iloc[-2]['close']
        
        # Moving averages
        if len(bars_df) >= 20:
            features['sma_20'] = bars_df['close'].tail(20).mean()
            features['sma_50'] = bars_df['close'].tail(50).mean() if len(bars_df) >= 50 else features['sma_20']
        
        # Volatility
        if len(bars_df) >= 20:
            features['volatility'] = bars_df['close'].tail(20).std()
        
        return features
    
    def compute_features_from_quote(self, quote) -> dict:
        """Compute features from quote data"""
        return {
            'bid': quote.bid,
            'ask': quote.ask,
            'spread': quote.spread,
            'mid': quote.mid,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size,
        }
    
    def generate_signal(self, features: dict, model_key: str) -> dict:
        """Generate signal from features using specified model"""
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        scaler = self.scalers.get(model_key)
        feature_list = self.features.get(model_key, [])
        
        # Create feature vector
        feature_values = []
        for feat in feature_list:
            if feat in features:
                feature_values.append(features[feat])
            else:
                feature_values.append(0)  # Missing feature
        
        if len(feature_values) == 0:
            return None
        
        # Scale features
        X = np.array(feature_values).reshape(1, -1)
        if scaler:
            X = scaler.transform(X)
        
        # Predict
        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
            
            return {
                'signal': 'BUY' if pred == 1 else 'SELL',
                'confidence': float(max(proba)),
                'model': model_key
            }
        except Exception as e:
            logger.error(f"Prediction error for {model_key}: {e}")
            return None
    
    async def run_once(self):
        """Generate signals once"""
        logger.info("\n" + "="*80)
        logger.info(f"LIVE SIGNAL GENERATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("="*80)
        
        # Fetch data
        logger.info("\n[1/3] Fetching live market data...")
        start = time.time()
        
        try:
            quote, bars = await self.fetch_live_data()
            elapsed = time.time() - start
            
            logger.info(f"  ✓ Data fetched in {elapsed*1000:.0f}ms")
            logger.info(f"    Quote: Bid={quote.bid:.2f}, Ask={quote.ask:.2f}, Spread={quote.spread:.2f}")
            logger.info(f"    Bars: {len(bars)} timeframes")
            
        except Exception as e:
            logger.error(f"  ✗ Data fetch failed: {e}")
            return
        
        # Generate signals for each timeframe
        logger.info("\n[2/3] Generating signals...")
        
        signals = {}
        
        for tf in ['1T', '5T', '15T', '30T']:
            if tf not in bars:
                continue
            
            # OHLCV signals
            bars_df = bars[tf]
            ohlcv_features = self.compute_features_from_bars(bars_df, tf)
            
            if ohlcv_features:
                signal = self.generate_signal(ohlcv_features, f'ohlcv_{tf}')
                if signal:
                    signals[f'ohlcv_{tf}'] = signal
            
            # Quote signals
            quote_features = self.compute_features_from_quote(quote)
            signal = self.generate_signal(quote_features, f'quote_{tf}')
            if signal:
                signals[f'quote_{tf}'] = signal
        
        # Aggregate signals
        logger.info("\n[3/3] Signal Summary:")
        
        buy_votes = sum(1 for s in signals.values() if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in signals.values() if s['signal'] == 'SELL')
        total_votes = len(signals)
        
        if total_votes == 0:
            logger.warning("  No signals generated")
            return
        
        # Final signal
        final_signal = 'BUY' if buy_votes > sell_votes else 'SELL'
        confidence = max(buy_votes, sell_votes) / total_votes
        
        logger.info(f"  Timeframes analyzed: {total_votes}")
        logger.info(f"  BUY votes: {buy_votes}")
        logger.info(f"  SELL votes: {sell_votes}")
        logger.info(f"\n  → FINAL SIGNAL: {final_signal} (confidence: {confidence:.1%})")
        
        # Display details
        logger.info("\n  Individual Signals:")
        for key, sig in sorted(signals.items()):
            logger.info(f"    {key:15s}: {sig['signal']:4s} ({sig['confidence']:.1%})")
        
        logger.info("\n" + "="*80)
        
    async def run_continuous(self, interval: int = 60):
        """Run signal generation continuously"""
        logger.info(f"Starting continuous signal generation (every {interval}s)")
        
        while True:
            try:
                await self.run_once()
                await asyncio.sleep(interval)
            except KeyboardInterrupt:
                logger.info("\nStopping signal generation...")
                break
            except Exception as e:
                logger.error(f"Error in continuous run: {e}")
                await asyncio.sleep(interval)


async def main():
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY not found in environment")
        return
    
    generator = LiveSignalGenerator(api_key)
    generator.load_models()
    
    # Run once or continuously
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        await generator.run_continuous()
    else:
        await generator.run_once()


if __name__ == "__main__":
    asyncio.run(main())
