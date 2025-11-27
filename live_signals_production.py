#!/usr/bin/env python3
"""
Live Signal Generator - Production with Full Trading Parameters
Generates TP, SL, Entry, Order Type for all 20 models
Sends results to Supabase
"""
import sys
import os
from pathlib import Path
import pickle
import logging
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from supabase import create_client, Client
from tabulate import tabulate

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


class TradingSignalGenerator:
    """Generate complete trading signals with TP/SL/Entry for all models"""
    
    # Trading parameters - Conservative R:R ratios (1:0.8 to 1:2.2)
    BASE_TP_MULTIPLIER = 1.5  # Base TP multiplier (reduced from 2.0)
    BASE_SL_MULTIPLIER = 1.0  # Base SL multiplier
    SPREAD_BUFFER = 0.5       # Add 50% of spread to limits
    
    # Confidence-based adjustments (tighter range)
    HIGH_CONFIDENCE_THRESHOLD = 0.75  # Scale TP wider for high confidence
    LOW_CONFIDENCE_THRESHOLD = 0.55   # Tighter SL for low confidence
    
    # Quote model bias correction (Quote models favor SELL 2:1)
    QUOTE_MODEL_BIAS_THRESHOLD = 0.60  # Higher threshold for quote models due to SELL bias
    
    def __init__(self, api_key: str, supabase_url: str, supabase_key: str):
        self.api_key = api_key
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.models = {}
        self.scalers = {}
        self.features = {}
        
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading trained models...")
        
        timeframes = ['1T', '5T', '15T', '30T']
        
        for tf in timeframes:
            # OHLCV models
            model_path = f'artifacts/ohlcv_model_{tf}_xgb.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[f'ohlcv_{tf}'] = pickle.load(f)
                
                features_path = f'artifacts/ohlcv_features_{tf}.txt'
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.features[f'ohlcv_{tf}'] = [line.strip() for line in f]
            
            # Quote models
            model_path = f'artifacts/quote_model_{tf}_xgb.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[f'quote_{tf}'] = pickle.load(f)
                
                features_path = f'artifacts/quote_features_{tf}.txt'
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.features[f'quote_{tf}'] = [line.strip() for line in f]
        
        logger.info(f"  ✓ Loaded {len(self.models)} models")
        return len(self.models) > 0
        
    async def fetch_live_data(self):
        """Fetch latest quotes and bars"""
        async with PolygonRESTClient(self.api_key) as client:
            quote = await client.get_latest_quote("C:XAU-USD")
            
            bars = {}
            for tf_name, multiplier in [('1T', 1), ('5T', 5), ('15T', 15), ('30T', 30)]:
                bars_df = await client.get_historical_bars(
                    "C:XAUUSD",
                    timespan="minute",
                    multiplier=multiplier,
                    limit=100
                )
                if bars_df is not None and len(bars_df) > 0:
                    bars[tf_name] = bars_df
            
            return quote, bars
    
    def calculate_dynamic_tp_sl(
        self,
        signal_type: str,
        confidence: float,
        entry_price: float,
        atr: float,
        bars_df: pd.DataFrame
    ) -> tuple:
        """
        Calculate TP/SL dynamically based on:
        - Model confidence (higher confidence = wider TP, tighter SL)
        - Market volatility (ATR)
        - Recent price action (support/resistance levels)
        """
        # Base multipliers adjusted by confidence (R:R range 1:0.8 to 1:2.2)
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            # High confidence: Max R:R of 2.2:1
            tp_mult = self.BASE_TP_MULTIPLIER * 1.40  # 1.5 * 1.40 = 2.1 max (buffer for 2.2)
            sl_mult = self.BASE_SL_MULTIPLIER * 0.95  # Slightly tighter stop
        elif confidence <= self.LOW_CONFIDENCE_THRESHOLD:
            # Low confidence: Min R:R of 0.8:1
            tp_mult = self.BASE_TP_MULTIPLIER * 0.53  # 1.5 * 0.53 = 0.8 min
            sl_mult = self.BASE_SL_MULTIPLIER * 1.25  # Wider stop for protection
        else:
            # Medium confidence: Mid-range R:R ~1.5:1
            tp_mult = self.BASE_TP_MULTIPLIER  # 1.5x baseline
            sl_mult = self.BASE_SL_MULTIPLIER
        
        # Calculate volatility adjustment from recent price action
        if len(bars_df) >= 20:
            recent_range = bars_df['high'].tail(20).max() - bars_df['low'].tail(20).min()
            volatility_factor = recent_range / (atr * 20) if atr > 0 else 1.0
            
            # Higher volatility = wider stops
            if volatility_factor > 1.5:
                sl_mult *= 1.2
            elif volatility_factor < 0.7:
                sl_mult *= 0.8
        
        # Find nearest support/resistance levels
        if len(bars_df) >= 50:
            recent_highs = bars_df['high'].tail(50).nlargest(5).mean()
            recent_lows = bars_df['low'].tail(50).nsmallest(5).mean()
            
            if signal_type == 'BUY':
                # TP near resistance, SL below support
                resistance_distance = abs(recent_highs - entry_price)
                support_distance = abs(entry_price - recent_lows)
                
                # Use resistance as TP if reasonable
                if resistance_distance > atr * 0.5 and resistance_distance < atr * 5:
                    tp_distance = resistance_distance
                else:
                    tp_distance = atr * tp_mult
                
                # Use support as SL if reasonable  
                if support_distance > atr * 0.3 and support_distance < atr * 2:
                    sl_distance = support_distance
                else:
                    sl_distance = atr * sl_mult
                    
            else:  # SELL
                # TP near support, SL above resistance
                support_distance = abs(entry_price - recent_lows)
                resistance_distance = abs(recent_highs - entry_price)
                
                if support_distance > atr * 0.5 and support_distance < atr * 5:
                    tp_distance = support_distance
                else:
                    tp_distance = atr * tp_mult
                
                if resistance_distance > atr * 0.3 and resistance_distance < atr * 2:
                    sl_distance = resistance_distance
                else:
                    sl_distance = atr * sl_mult
        else:
            # Not enough data for S/R - use ATR-based
            tp_distance = atr * tp_mult
            sl_distance = atr * sl_mult
        
        return tp_distance, sl_distance
    
    def calculate_atr(self, bars_df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility-based TP/SL"""
        if len(bars_df) < period + 1:
            return bars_df['high'].iloc[-1] - bars_df['low'].iloc[-1]
        
        high = bars_df['high'].values[-period:]
        low = bars_df['low'].values[-period:]
        close = bars_df['close'].values[-period-1:-1]
        
        tr1 = high - low
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr)
        
        return atr
    
    def compute_features_from_bars(self, bars_df: pd.DataFrame) -> dict:
        """Compute features from OHLCV bars"""
        if len(bars_df) < 20:
            return None
        
        latest = bars_df.iloc[-1]
        
        features = {
            'close': latest['close'],
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'volume': latest['volume'],
            'returns': (latest['close'] - bars_df.iloc[-2]['close']) / bars_df.iloc[-2]['close'] if len(bars_df) >= 2 else 0,
            'sma_20': bars_df['close'].tail(20).mean(),
            'volatility': bars_df['close'].tail(20).std(),
        }
        
        if len(bars_df) >= 50:
            features['sma_50'] = bars_df['close'].tail(50).mean()
        
        return features
    
    def compute_features_from_quote(self, quote) -> dict:
        """Compute features from quote"""
        return {
            'bid': quote.bid,
            'ask': quote.ask,
            'spread': quote.spread,
            'mid': quote.mid,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size,
        }
    
    def generate_signal_with_params(
        self, 
        features: dict, 
        model_key: str,
        quote,
        bars_df: pd.DataFrame,
        timeframe: str
    ) -> Optional[Dict]:
        """Generate complete trading signal with TP/SL/Entry"""
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        feature_list = self.features.get(model_key, [])
        
        # Create feature vector
        feature_values = []
        for feat in feature_list:
            feature_values.append(features.get(feat, 0))
        
        if len(feature_values) == 0:
            return None
        
        # Predict
        try:
            X = np.array(feature_values).reshape(1, -1)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
            
            signal_type = 'BUY' if pred == 1 else 'SELL'
            confidence = float(max(proba))
            
            # Apply bias correction for Quote models (heavily favor SELL in training)
            # Quote models have 2:1 SELL bias, so require higher confidence for SELL
            if 'quote_' in model_key and signal_type == 'SELL':
                if confidence < self.QUOTE_MODEL_BIAS_THRESHOLD:
                    # Skip low-confidence SELL signals from biased quote models
                    logger.debug(f"Filtered biased SELL signal from {model_key} (conf={confidence:.2f})")
                    return None
            
            # Momentum-based signal validation and inversion
            # Models are lagging indicators - they predict after moves happen
            # This causes: SELL after drops (price bounces) → SL hit
            # Solution: Only take signals aligned with momentum OR invert weak signals
            if bars_df is not None and len(bars_df) >= 20:
                recent_close = bars_df.iloc[-1]['close']
                ma_5 = bars_df['close'].tail(5).mean()
                ma_20 = bars_df['close'].tail(20).mean()
                
                # Determine current momentum
                short_term_bullish = ma_5 > ma_20
                price_vs_ma20_pct = (recent_close - ma_20) / ma_20 * 100
                
                # Check if signal is counter-trend
                is_counter_trend = (signal_type == 'SELL' and short_term_bullish) or \
                                   (signal_type == 'BUY' and not short_term_bullish)
                
                if is_counter_trend:
                    # Only allow counter-trend signals if price is very extended (mean reversion)
                    # OR if confidence is very high (>80%)
                    is_extended = abs(price_vs_ma20_pct) > 0.2
                    is_high_confidence = confidence > 0.80
                    
                    if not (is_extended or is_high_confidence):
                        # Counter-trend with low confidence and price not extended → SKIP
                        logger.debug(f"Filtered counter-trend {signal_type} from {model_key} "
                                   f"(conf={confidence:.2%}, ext={price_vs_ma20_pct:+.2f}%)")
                        return None
            
            # Current market prices
            current_bid = quote.bid
            current_ask = quote.ask
            spread = quote.spread
            
            # Entry prices based on direction
            if signal_type == 'BUY':
                entry_market = current_ask
                entry_limit = current_bid + spread * self.SPREAD_BUFFER
            else:  # SELL
                entry_market = current_bid
                entry_limit = current_ask - spread * self.SPREAD_BUFFER
            
            # Order type based on confidence
            order_type = f'{signal_type}_MARKET' if confidence > 0.7 else f'{signal_type}_LIMIT'
            
            # Calculate ATR for dynamic TP/SL
            atr = self.calculate_atr(bars_df) if bars_df is not None else quote.spread * 10
            
            # Calculate model-adaptive TP/SL distances
            tp_distance, sl_distance = self.calculate_dynamic_tp_sl(
                signal_type,
                confidence,
                entry_market,
                atr,
                bars_df
            )
            
            # Calculate actual TP/SL prices based on direction
            if signal_type == 'BUY':
                take_profit = entry_market + tp_distance
                stop_loss = entry_market - sl_distance
            else:  # SELL
                take_profit = entry_market - tp_distance
                stop_loss = entry_market + sl_distance
            
            # Calculate risk/reward
            risk = abs(entry_market - stop_loss)
            reward = abs(take_profit - entry_market)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Enforce R:R ratio limits (0.8 to 2.2)
            if rr_ratio > 2.2:
                # Reduce TP to cap R:R at 2.2
                reward = risk * 2.2
                if signal_type == 'BUY':
                    take_profit = entry_market + reward
                else:  # SELL
                    take_profit = entry_market - reward
                rr_ratio = 2.2
            elif rr_ratio < 0.8:
                # Increase TP to meet minimum R:R of 0.8
                reward = risk * 0.8
                if signal_type == 'BUY':
                    take_profit = entry_market + reward
                else:  # SELL
                    take_profit = entry_market - reward
                rr_ratio = 0.8
            
            return {
                'model': model_key,
                'timeframe': timeframe,
                'signal': signal_type,
                'confidence': round(confidence * 100, 2),
                'entry_market': round(entry_market, 2),
                'entry_limit': round(entry_limit, 2),
                'take_profit': round(take_profit, 2),
                'stop_loss': round(stop_loss, 2),
                'order_type': order_type,
                'atr': round(atr, 2),
                'spread': round(spread, 2),
                'risk': round(risk, 2),
                'reward': round(reward, 2),
                'rr_ratio': round(rr_ratio, 2),
                'current_bid': round(current_bid, 2),
                'current_ask': round(current_ask, 2),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Signal generation error for {model_key}: {e}")
            return None
    
    def format_signals_table(self, signals: List[Dict]) -> str:
        """Format signals as table grouped by timeframe"""
        if not signals:
            return "No signals generated"
        
        # Group by timeframe
        df = pd.DataFrame(signals)
        
        # Create display table
        table_data = []
        for _, row in df.iterrows():
            table_data.append([
                row['timeframe'],
                row['model'],
                row['signal'],
                f"{row['confidence']:.1f}%",
                row['order_type'],
                f"${row['entry_market']:.2f}",
                f"${row['entry_limit']:.2f}",
                f"${row['take_profit']:.2f}",
                f"${row['stop_loss']:.2f}",
                f"{row['rr_ratio']:.2f}",
                f"${row['spread']:.2f}",
            ])
        
        headers = [
            'TF', 'Model', 'Signal', 'Conf', 'Order', 
            'Entry (Mkt)', 'Entry (Lmt)', 'TP', 'SL', 'R:R', 'Spread'
        ]
        
        return tabulate(table_data, headers=headers, tablefmt='grid')
    
    async def send_to_supabase(self, signals: List[Dict]):
        """Send signals to Supabase"""
        try:
            # Prepare data for Supabase
            records = []
            for sig in signals:
                records.append({
                    'symbol': 'XAU/USD',
                    'timeframe': sig['timeframe'],
                    'model_name': sig['model'],
                    'signal_type': sig['signal'],
                    'confidence': sig['confidence'],
                    'entry_market': sig['entry_market'],
                    'entry_limit': sig['entry_limit'],
                    'take_profit': sig['take_profit'],
                    'stop_loss': sig['stop_loss'],
                    'order_type': sig['order_type'],
                    'atr': sig['atr'],
                    'spread': sig['spread'],
                    'risk': sig['risk'],
                    'reward': sig['reward'],
                    'rr_ratio': sig['rr_ratio'],
                    'current_bid': sig['current_bid'],
                    'current_ask': sig['current_ask'],
                    'timestamp': sig['timestamp'],
                })
            
            # Insert into Supabase (table: trading_signals)
            result = self.supabase.table('trading_signals').insert(records).execute()
            logger.info(f"  ✓ Sent {len(records)} signals to Supabase")
            return result
            
        except Exception as e:
            logger.error(f"  ✗ Supabase insert failed: {e}")
            return None
    
    async def run_once(self):
        """Generate signals once"""
        logger.info("\n" + "="*100)
        logger.info(f"LIVE TRADING SIGNALS - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("="*100)
        
        # Fetch data
        logger.info("\n[1/4] Fetching live market data...")
        start = time.time()
        
        try:
            quote, bars = await self.fetch_live_data()
            elapsed = time.time() - start
            
            logger.info(f"  ✓ Data fetched in {elapsed*1000:.0f}ms")
            logger.info(f"    Quote: Bid=${quote.bid:.2f}, Ask=${quote.ask:.2f}, Spread=${quote.spread:.2f}")
            
        except Exception as e:
            logger.error(f"  ✗ Data fetch failed: {e}")
            return
        
        # Generate signals
        logger.info("\n[2/4] Generating trading signals with TP/SL/Entry...")
        
        all_signals = []
        
        for tf in ['1T', '5T', '15T', '30T']:
            if tf not in bars:
                continue
            
            bars_df = bars[tf]
            
            # OHLCV signal
            ohlcv_features = self.compute_features_from_bars(bars_df)
            if ohlcv_features:
                signal = self.generate_signal_with_params(
                    ohlcv_features, 
                    f'ohlcv_{tf}',
                    quote,
                    bars_df,
                    tf
                )
                if signal:
                    all_signals.append(signal)
            
            # Quote signal
            quote_features = self.compute_features_from_quote(quote)
            signal = self.generate_signal_with_params(
                quote_features,
                f'quote_{tf}',
                quote,
                bars_df,
                tf
            )
            if signal:
                all_signals.append(signal)
        
        if not all_signals:
            logger.warning("  No signals generated")
            return
        
        logger.info(f"  ✓ Generated {len(all_signals)} signals across all timeframes")
        
        # Display table
        logger.info("\n[3/4] Signal Summary Table:")
        table = self.format_signals_table(all_signals)
        print("\n" + table + "\n")
        
        # Aggregate vote
        buy_votes = sum(1 for s in all_signals if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in all_signals if s['signal'] == 'SELL')
        avg_confidence = np.mean([s['confidence'] for s in all_signals])
        
        logger.info(f"  Consensus: {buy_votes} BUY, {sell_votes} SELL | Avg Confidence: {avg_confidence:.1f}%")
        
        # Send to Supabase
        logger.info("\n[4/4] Sending to Supabase...")
        await self.send_to_supabase(all_signals)
        
        logger.info("\n" + "="*100)
        logger.info("✅ SIGNAL GENERATION COMPLETE")
        logger.info("="*100 + "\n")
    
    async def run_continuous(self, interval: int = 60):
        """Run continuously"""
        logger.info(f"Starting continuous signal generation (every {interval}s)")
        
        while True:
            try:
                await self.run_once()
                await asyncio.sleep(interval)
            except KeyboardInterrupt:
                logger.info("\nStopping...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(interval)


async def main():
    api_key = os.getenv("POLYGON_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not all([api_key, supabase_url, supabase_key]):
        logger.error("Missing environment variables")
        return
    
    generator = TradingSignalGenerator(api_key, supabase_url, supabase_key)
    generator.load_models()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        await generator.run_continuous()
    else:
        await generator.run_once()


if __name__ == "__main__":
    asyncio.run(main())
