"""
Live Signal Generator - Main Orchestrator
==========================================

Real-time signal generation pipeline:
Polygon.io → Feature Engine → ONNX Models → Supabase
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import List

import pandas as pd

from live_signal_engine import SignalGenerator, Signal, Config, AggBar, Quote
from polygon_connector import PolygonWebSocketClient
from supabase_store import SupabaseSignalStore, SignalBroadcaster

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class LiveSignalOrchestrator:
    """Orchestrates real-time signal generation end-to-end."""
    
    def __init__(self):
        """Initialize orchestrator."""
        logger.info("=" * 100)
        logger.info("LIVE SIGNAL GENERATOR - STARTING UP")
        logger.info("=" * 100)
        
        # Validate config
        Config.validate()
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        self.polygon_client = None
        self.supabase_store = None
        self.broadcaster = SignalBroadcaster()
        
        # Statistics
        self.stats = {
            'bars_processed': 0,
            'signals_generated': 0,
            'signals_stored': 0,
            'start_time': datetime.now(timezone.utc),
        }
        
        # Try to connect to Supabase
        try:
            self.supabase_store = SupabaseSignalStore()
            logger.info("✓ Supabase store connected")
        except Exception as e:
            logger.warning(f"Supabase not available: {e}")
            self.supabase_store = None
        
        logger.info("✓ Orchestrator initialized")
    
    def on_polygon_agg(self, agg: AggBar) -> None:
        """Handle incoming Polygon minute aggregate."""
        self.stats['bars_processed'] += 1
        
        try:
            # Generate signals
            signals = self.signal_generator.process_polygon_agg(agg)
            
            if signals:
                self.stats['signals_generated'] += len(signals)
                self._handle_signals(signals)
        
        except Exception as e:
            logger.error(f"Error processing aggregate: {e}", exc_info=True)
    
    def on_polygon_quote(self, quote: Quote) -> None:
        """Handle incoming Polygon quote."""
        try:
            self.signal_generator.process_quote(quote)
        except Exception as e:
            logger.error(f"Error processing quote: {e}")
    
    def _handle_signals(self, signals: List[Signal]) -> None:
        """Process generated signals."""
        for signal in signals:
            # Log signal
            logger.info(
                f"📊 SIGNAL: {signal.symbol} | {signal.timeframe} | "
                f"Direction: {signal.signal:+d} | Confidence: {signal.confidence:.2%}"
            )
            
            # Broadcast
            self.broadcaster.broadcast(signal)
            
            # Store in Supabase
            if self.supabase_store:
                if self.supabase_store.store_signal(signal):
                    self.stats['signals_stored'] += 1
                else:
                    logger.warning(f"Failed to store signal to Supabase")
    
    async def start_live_stream(self) -> None:
        """Start receiving live data from Polygon."""
        logger.info("Starting Polygon WebSocket stream...")
        
        # Create Polygon client
        self.polygon_client = PolygonWebSocketClient(
            api_key=Config.POLYGON_API_KEY,
            symbol=Config.SYMBOL,
            on_agg_callback=self.on_polygon_agg,
            on_quote_callback=self.on_polygon_quote,
        )
        
        # Connect (blocking)
        try:
            await self.polygon_client.connect()
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            self._print_summary()
    
    def _print_summary(self) -> None:
        """Print statistics summary."""
        elapsed = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds()
        
        logger.info("\n" + "=" * 100)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 100)
        logger.info(f"Duration: {elapsed:.1f} seconds")
        logger.info(f"Bars processed: {self.stats['bars_processed']}")
        logger.info(f"Signals generated: {self.stats['signals_generated']}")
        logger.info(f"Signals stored: {self.stats['signals_stored']}")
        if elapsed > 0:
            logger.info(f"Throughput: {self.stats['bars_processed'] / elapsed:.1f} bars/sec")
        logger.info("=" * 100 + "\n")


def standalone_test():
    """Test signal generation with mock data."""
    logger.info("STANDALONE TEST MODE - No Polygon connection")
    logger.info("=" * 100)
    
    from live_signal_engine import AggBar
    
    # Create signal generator
    signal_gen = SignalGenerator()
    
    # Create mock bars
    base_time = pd.Timestamp('2024-01-01 00:00', tz='UTC')
    
    logger.info("Generating synthetic market data...")
    
    for i in range(100):
        timestamp = base_time + pd.Timedelta(minutes=i)
        close_price = 2000 + (i % 10) * 5  # Synthetic price movement
        
        bar = AggBar(
            symbol="XAUUSD",
            o=close_price,
            h=close_price + 2,
            l=close_price - 2,
            c=close_price,
            v=1000,
            start_ts=timestamp - pd.Timedelta(minutes=1),
            end_ts=timestamp,
        )
        
        # Process
        signals = signal_gen.process_polygon_agg(bar)
        
        if signals:
            for sig in signals:
                logger.info(
                    f"  ✓ {sig.symbol} {sig.timeframe} "
                    f"signal={sig.signal:+d} conf={sig.confidence:.2f}"
                )
    
    logger.info("=" * 100)
    logger.info(f"Total signals generated: {len(signal_gen.signals_history)}")
    logger.info("=" * 100 + "\n")


async def main():
    """Main entry point."""
    # Parse arguments
    test_mode = "--test" in sys.argv
    
    if test_mode:
        standalone_test()
    else:
        # Live mode
        orchestrator = LiveSignalOrchestrator()
        
        try:
            await orchestrator.start_live_stream()
        except KeyboardInterrupt:
            logger.info("\nShutdown...")
            orchestrator._print_summary()


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        sys.exit(1)
    
    # Run
    asyncio.run(main())
