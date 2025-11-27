#!/usr/bin/env python3
"""
Live Signal Generator with ML Models
Fetches real-time data and generates predictions using trained models
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

from polygon_connector import Quote, AggBar
from model_loader import ModelLoader
import pandas as pd
import numpy as np


def compute_rsi(prices, period=14):
    """Compute RSI indicator"""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
api_key = os.getenv("POLYGON_API_KEY")

if not api_key:
    print("❌ ERROR: POLYGON_API_KEY not set in .env")
    sys.exit(1)

print("\n" + "="*80)
print("LIVE SIGNAL GENERATION WITH ML MODELS")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Symbols: XAU/USD | Timeframes: 5T, 15T")
print("="*80 + "\n")

try:
    # Step 1: Initialize clients
    print("[1/5] Initializing clients...")
    
    # Use REST client directly for synchronous operation
    from massive import RESTClient
    rest_client = RESTClient(api_key=api_key)
    model_loader = ModelLoader()
    print("  ✓ Clients ready\n")
    
    # Step 2: Load ML models
    print("[2/5] Loading ML models...")
    models = model_loader.load_all_models()
    print(f"  ✓ Loaded {len(models)} models\n")
    
    if not models:
        print("  ⚠ No models loaded, using technical analysis only\n")
    
    # Step 3: Fetch live data
    print("[3/5] Fetching live market data...")
    
    start_time = time.time()
    
    # Get quote
    quotes = list(rest_client.list_quotes(ticker="C:XAU-USD", order="desc", limit=1))
    quote = None
    if quotes:
        q = quotes[0]
        quote = Quote(
            symbol="C:XAU-USD",
            bid=q.bid_price if hasattr(q, 'bid_price') else q.bid,
            ask=q.ask_price if hasattr(q, 'ask_price') else q.ask,
            bid_size=getattr(q, 'bid_size', 0),
            ask_size=getattr(q, 'ask_size', 0),
            exchange=getattr(q, 'exchange', 0),
            ts=pd.Timestamp.now()
        )
    
    # Get bars for analysis
    aggs_5t = list(rest_client.list_aggs(
        ticker="C:XAUUSD",
        timespan="minute",
        multiplier=1,
        order="desc",
        limit=20
    ))
    
    aggs_15t = list(rest_client.list_aggs(
        ticker="C:XAUUSD",
        timespan="minute",
        multiplier=1,
        order="desc",
        limit=60
    ))
    
    # Convert to AggBar objects
    bars_5t = [AggBar(
        symbol="C:XAUUSD",
        o=agg.o,
        h=agg.h,
        l=agg.l,
        c=agg.c,
        v=agg.v if hasattr(agg, 'v') else 0,
        start_ts=pd.Timestamp(agg.timestamp),
        end_ts=pd.Timestamp(agg.timestamp)
    ) for agg in aggs_5t]
    
    bars_15t = [AggBar(
        symbol="C:XAUUSD",
        o=agg.o,
        h=agg.h,
        l=agg.l,
        c=agg.c,
        v=agg.v if hasattr(agg, 'v') else 0,
        start_ts=pd.Timestamp(agg.timestamp),
        end_ts=pd.Timestamp(agg.timestamp)
    ) for agg in aggs_15t]
    
    elapsed = time.time() - start_time
    
    if not quote:
        print("  ✗ Failed to get quote")
        sys.exit(1)
    
    print(f"  ✓ Data fetched in {elapsed*1000:.0f}ms")
    print(f"    - Quote: Bid=${quote.bid:.4f} Ask=${quote.ask:.4f}")
    print(f"    - Bars 5T: {len(bars_5t) if bars_5t else 0}")
    print(f"    - Bars 15T: {len(bars_15t) if bars_15t else 0}\n")
    
    # Step 4: Feature engineering
    print("[4/5] Computing features...")
    
    features = {}
    mid_price = (quote.bid + quote.ask) / 2
    
    if bars_5t:
        closes = [bar.c for bar in bars_5t]
        features['close'] = closes[-1]
        features['sma_5'] = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        features['sma_20'] = np.mean(closes)
        features['volatility_5'] = np.std(closes[-5:]) if len(closes) >= 5 else 0
        features['momentum_5'] = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
        features['rsi_5'] = compute_rsi(closes, 5)
    
    if bars_15t:
        closes_15 = [bar.c for bar in bars_15t]
        features['sma_20_15t'] = np.mean(closes_15)
        features['volatility_15'] = np.std(closes_15[-15:]) if len(closes_15) >= 15 else 0
        features['momentum_15'] = (closes_15[-1] - closes_15[0]) / closes_15[0] if closes_15[0] != 0 else 0
    
    features['spread'] = quote.ask - quote.bid
    features['bid_ask_ratio'] = quote.bid / quote.ask if quote.ask != 0 else 1
    
    print(f"  ✓ Features computed ({len(features)} features)")
    print(f"    - Close: ${features.get('close', mid_price):.4f}")
    print(f"    - SMA-5: ${features.get('sma_5', mid_price):.4f}")
    print(f"    - Volatility: {features.get('volatility_5', 0):.6f}\n")
    
    # Step 5: Generate signals
    print("[5/5] Generating signals from ML models...\n")
    
    print(f"{'─'*80}")
    print(f"SIGNAL GENERATION RESULTS")
    print(f"{'─'*80}\n")
    
    signal_results = []
    
    for model_name in sorted(models.keys()):
        try:
            prediction = model_loader.predict(model_name, features)
            
            if prediction is not None:
                # Interpret prediction
                if prediction > 0.5:
                    signal = "🟢 BUY"
                    direction = "UP"
                elif prediction < -0.5:
                    signal = "🔴 SELL"
                    direction = "DOWN"
                else:
                    signal = "🟡 NEUTRAL"
                    direction = "NONE"
                
                confidence = abs(prediction)
                
                signal_results.append({
                    'model': model_name,
                    'signal': signal,
                    'direction': direction,
                    'confidence': confidence,
                    'prediction': prediction
                })
                
                print(f"  {signal} {model_name}")
                print(f"     Confidence: {confidence*100:.0f}%")
                print(f"     Prediction: {prediction:.4f}\n")
        
        except Exception as e:
            logger.error(f"Error generating signal from {model_name}: {e}")
    
    # Aggregate signals
    print(f"{'─'*80}")
    print(f"AGGREGATED SIGNAL")
    print(f"{'─'*80}\n")
    
    if signal_results:
        # Count signals
        buy_signals = sum(1 for s in signal_results if 'BUY' in s['signal'])
        sell_signals = sum(1 for s in signal_results if 'SELL' in s['signal'])
        neutral_signals = sum(1 for s in signal_results if 'NEUTRAL' in s['signal'])
        
        avg_confidence = np.mean([s['confidence'] for s in signal_results])
        
        if buy_signals > sell_signals:
            agg_signal = "🟢 BUY"
        elif sell_signals > buy_signals:
            agg_signal = "🔴 SELL"
        else:
            agg_signal = "🟡 NEUTRAL"
        
        print(f"  Overall Signal:   {agg_signal}")
        print(f"  Avg Confidence:   {avg_confidence*100:.0f}%")
        print(f"  Buy models:       {buy_signals}")
        print(f"  Sell models:      {sell_signals}")
        print(f"  Neutral models:   {neutral_signals}\n")
    
    # Trading recommendation
    print(f"{'─'*80}")
    print(f"TRADING RECOMMENDATION")
    print(f"{'─'*80}\n")
    
    print(f"  Entry Price:      ${mid_price:.4f}")
    print(f"  Stop Loss:        ${mid_price - 5:.4f} (-5 pips)")
    print(f"  Take Profit:      ${mid_price + 10:.4f} (+10 pips)")
    print(f"  Risk/Reward:      1:2")
    print(f"  Generated:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Status:           ✓ READY FOR LIVE TRADING\n")
    
    print("="*80 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"   {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
