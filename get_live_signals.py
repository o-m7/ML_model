#!/usr/bin/env python3
"""
live_signal_generator.py
========================
Production-grade live signal generation for XAU/USD.

Performance Targets:
- Data fetch: <50ms (P95)
- Signal generation: <10ms
- Total latency: <60ms (P95)

Features:
- Parallel quote + bar fetching via WebSocket/REST hybrid
- Multi-timeframe technical analysis
- Supabase signal storage
- Real-time performance monitoring
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from polygon_connector import (
    HybridPolygonClient,
    Quote,
    AggBar
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Vectorized technical indicators for speed"""
    
    @staticmethod
    def sma(series: np.ndarray, period: int) -> float:
        """Simple moving average"""
        if len(series) < period:
            return np.mean(series)
        return np.mean(series[:period])
    
    @staticmethod
    def ema(series: np.ndarray, period: int) -> float:
        """Exponential moving average"""
        if len(series) < period:
            return np.mean(series)
        
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(series[:period], weights, mode='valid')[0]
    
    @staticmethod
    def atr(bars: List[AggBar], period: int = 14) -> float:
        """Average True Range"""
        if len(bars) < period + 1:
            return 0.0
        
        trs = []
        for i in range(min(period, len(bars) - 1)):
            h = bars[i].h
            l = bars[i].l
            prev_close = bars[i + 1].c
            
            tr = max(
                h - l,
                abs(h - prev_close),
                abs(l - prev_close)
            )
            trs.append(tr)
        
        return np.mean(trs) if trs else 0.0
    
    @staticmethod
    def rsi(closes: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes[:period + 1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(closes: np.ndarray) -> Dict[str, float]:
        """MACD indicator"""
        if len(closes) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema12 = TechnicalIndicators.ema(closes, 12)
        ema26 = TechnicalIndicators.ema(closes, 26)
        macd = ema12 - ema26
        
        # Signal line (9-period EMA of MACD)
        # For simplicity, using SMA as approximation
        signal = macd * 0.9  # Simplified
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }


class SignalGenerator:
    """
    Institutional-grade signal generator.
    
    Strategy:
    - Multi-timeframe trend alignment
    - Momentum confirmation
    - Volatility-adjusted targets
    - Confidence scoring
    """
    
    def __init__(self):
        self.min_bars_required = 100
    
    def generate(
        self,
        quote: Quote,
        minute_bars: List[AggBar],
        second_bars: Optional[List[AggBar]] = None
    ) -> Dict:
        """
        Generate trading signal with confidence score.
        
        Returns:
            {
                'signal': 'BUY'|'SELL'|'NEUTRAL',
                'confidence': float (0-1),
                'entry': float,
                'stop_loss': float,
                'take_profit': float,
                'risk_reward': float,
                'indicators': dict,
                'timestamp': pd.Timestamp
            }
        """
        start = time.perf_counter()
        
        if len(minute_bars) < 20:
            logger.warning(f"Insufficient bars: {len(minute_bars)}")
            return self._neutral_signal(quote)
        
        # Extract closes
        closes = np.array([bar.c for bar in minute_bars])
        
        # Technical indicators
        sma_20 = TechnicalIndicators.sma(closes, 20)
        sma_50 = TechnicalIndicators.sma(closes, 50) if len(closes) >= 50 else sma_20
        ema_9 = TechnicalIndicators.ema(closes, 9)
        rsi = TechnicalIndicators.rsi(closes, 14)
        atr = TechnicalIndicators.atr(minute_bars, 14)
        macd_data = TechnicalIndicators.macd(closes)
        
        # Current price
        current_price = quote.mid
        
        # Signal logic
        signals = []
        
        # 1. Trend following
        if current_price > sma_20 > sma_50:
            signals.append(('BUY', 0.3, 'Uptrend'))
        elif current_price < sma_20 < sma_50:
            signals.append(('SELL', 0.3, 'Downtrend'))
        
        # 2. Momentum
        if ema_9 > sma_20:
            signals.append(('BUY', 0.2, 'Momentum up'))
        elif ema_9 < sma_20:
            signals.append(('SELL', 0.2, 'Momentum down'))
        
        # 3. RSI
        if rsi < 30:
            signals.append(('BUY', 0.2, 'Oversold'))
        elif rsi > 70:
            signals.append(('SELL', 0.2, 'Overbought'))
        elif 40 < rsi < 60:
            signals.append(('NEUTRAL', 0.1, 'RSI neutral'))
        
        # 4. MACD
        if macd_data['histogram'] > 0:
            signals.append(('BUY', 0.3, 'MACD bullish'))
        elif macd_data['histogram'] < 0:
            signals.append(('SELL', 0.3, 'MACD bearish'))
        
        # Aggregate signals
        buy_score = sum(conf for sig, conf, _ in signals if sig == 'BUY')
        sell_score = sum(conf for sig, conf, _ in signals if sig == 'SELL')
        
        # Determine final signal
        if buy_score > sell_score and buy_score >= 0.5:
            signal = 'BUY'
            confidence = min(buy_score, 1.0)
        elif sell_score > buy_score and sell_score >= 0.5:
            signal = 'SELL'
            confidence = min(sell_score, 1.0)
        else:
            signal = 'NEUTRAL'
            confidence = 0.3
        
        # Calculate levels (ATR-based)
        atr_multiplier_sl = 1.5
        atr_multiplier_tp = 3.0
        
        if signal == 'BUY':
            entry = quote.ask
            stop_loss = entry - (atr * atr_multiplier_sl)
            take_profit = entry + (atr * atr_multiplier_tp)
        elif signal == 'SELL':
            entry = quote.bid
            stop_loss = entry + (atr * atr_multiplier_sl)
            take_profit = entry - (atr * atr_multiplier_tp)
        else:
            entry = quote.mid
            stop_loss = entry - (atr * atr_multiplier_sl)
            take_profit = entry + (atr * atr_multiplier_tp)
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Execution time
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            'signal': signal,
            'confidence': round(confidence, 3),
            'entry': round(entry, 4),
            'stop_loss': round(stop_loss, 4),
            'take_profit': round(take_profit, 4),
            'risk_reward': round(risk_reward, 2),
            'indicators': {
                'sma_20': round(sma_20, 4),
                'sma_50': round(sma_50, 4),
                'ema_9': round(ema_9, 4),
                'rsi': round(rsi, 2),
                'atr': round(atr, 4),
                'macd': round(macd_data['macd'], 4),
                'macd_signal': round(macd_data['signal'], 4),
                'macd_histogram': round(macd_data['histogram'], 4)
            },
            'spread_bps': round(quote.spread_bps, 2),
            'timestamp': quote.ts,
            'generation_ms': round(elapsed_ms, 2),
            'reasons': [reason for _, _, reason in signals if _ == signal]
        }
    
    def _neutral_signal(self, quote: Quote) -> Dict:
        """Return neutral signal"""
        return {
            'signal': 'NEUTRAL',
            'confidence': 0.0,
            'entry': quote.mid,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'risk_reward': 0.0,
            'indicators': {},
            'spread_bps': quote.spread_bps,
            'timestamp': quote.ts,
            'generation_ms': 0.0,
            'reasons': ['Insufficient data']
        }


class SupabaseLogger:
    """Log signals to Supabase for monitoring"""
    
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
        self.table = "live_signals"
    
    async def log_signal(self, signal: Dict):
        """Insert signal into Supabase"""
        try:
            data = {
                'timestamp': signal['timestamp'].isoformat(),
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'entry': signal['entry'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'risk_reward': signal['risk_reward'],
                'indicators': signal['indicators'],
                'spread_bps': signal['spread_bps'],
                'generation_ms': signal['generation_ms']
            }
            
            result = self.client.table(self.table).insert(data).execute()
            logger.debug(f"Signal logged to Supabase: {signal['signal']}")
        
        except Exception as e:
            logger.error(f"Supabase logging error: {e}")


async def main():
    """Main live signal generation loop"""
    
    # Load config
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not api_key:
        logger.error("POLYGON_API_KEY not set in .env")
        sys.exit(1)
    
    # Initialize clients
    logger.info("="*80)
    logger.info("LIVE SIGNAL GENERATOR - XAU/USD")
    logger.info("="*80)
    
    polygon = HybridPolygonClient(
        api_key=api_key,
        ws_symbol="XAU/USD",
        rest_quote_symbol="C:XAU-USD",
        rest_bar_symbol="C:XAUUSD"
    )
    
    signal_gen = SignalGenerator()
    
    # Optional Supabase logging
    supabase_logger = None
    if supabase_url and supabase_key:
        supabase_logger = SupabaseLogger(supabase_url, supabase_key)
        logger.info("✓ Supabase logging enabled")
    
    # Start WebSocket streaming
    logger.info("Starting WebSocket streams...")
    await polygon.start_streaming(
        quotes=True,
        second_bars=True,
        minute_bars=True
    )
    logger.info("✓ Streaming active")
    
    # Fetch historical context (optional - for initial features)
    logger.info("Fetching historical context (50k minute bars)...")
    historical = await polygon.fetch_historical_context(
        quotes=0,  # Skip quotes for speed
        minute_bars=50000,
        second_bars=0
    )
    logger.info(f"✓ Historical: {len(historical['minute_bars'])} minute bars")
    
    # Main loop
    logger.info("\n" + "="*80)
    logger.info("LIVE SIGNAL GENERATION ACTIVE")
    logger.info("="*80 + "\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            loop_start = time.perf_counter()
            
            # Get live data
            market_data = await polygon.get_live_market_data()
            
            quote = market_data['quote']
            minute_bars = market_data['minute_bars']
            second_bars = market_data['second_bars']
            latency_stats = market_data['latency_stats']
            
            if not quote:
                logger.warning("No quote available, retrying...")
                await asyncio.sleep(1)
                continue
            
            # Generate signal
            signal = signal_gen.generate(quote, minute_bars, second_bars)
            
            # Calculate total latency
            total_latency_ms = (time.perf_counter() - loop_start) * 1000
            
            # Display signal
            logger.info(f"\n{'─'*80}")
            logger.info(f"[#{iteration}] SIGNAL GENERATED")
            logger.info(f"{'─'*80}")
            logger.info(f"  Time:        {signal['timestamp']}")
            logger.info(f"  Signal:      {signal['signal']} ({signal['confidence']*100:.0f}% confidence)")
            logger.info(f"  Entry:       ${signal['entry']:.4f}")
            logger.info(f"  Stop Loss:   ${signal['stop_loss']:.4f}")
            logger.info(f"  Take Profit: ${signal['take_profit']:.4f}")
            logger.info(f"  Risk/Reward: 1:{signal['risk_reward']:.2f}")
            logger.info(f"  Spread:      {signal['spread_bps']:.2f} bps")
            logger.info(f"\n  INDICATORS:")
            for key, val in signal['indicators'].items():
                logger.info(f"    {key.upper()}: {val}")
            logger.info(f"\n  REASONS:")
            for reason in signal['reasons']:
                logger.info(f"    • {reason}")
            logger.info(f"\n  PERFORMANCE:")
            logger.info(f"    Signal gen:  {signal['generation_ms']:.2f}ms")
            logger.info(f"    Total loop:  {total_latency_ms:.2f}ms")
            logger.info(f"    Minute bars: {len(minute_bars)}")
            logger.info(f"    Second bars: {len(second_bars)}")
            
            if latency_stats:
                logger.info(f"    WS latency (P50): {latency_stats.get('p50_ms', 0):.1f}ms")
                logger.info(f"    WS latency (P95): {latency_stats.get('p95_ms', 0):.1f}ms")
            
            logger.info(f"{'─'*80}\n")
            
            # Log to Supabase
            if supabase_logger:
                await supabase_logger.log_signal(signal)
            
            # Wait before next signal (e.g., every 5 seconds)
            await asyncio.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("\n\nShutting down...")
    
    finally:
        await polygon.stop()
        logger.info("✓ Disconnected")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")