#!/usr/bin/env python3
"""
Comprehensive test of Polygon data sources
- WebSocket: Real-time CurrencyAgg and ForexQuote
- REST API: Quotes, Previous bar, Today's bar
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from polygon_connector import PolygonWebSocketClient, PolygonRESTClient
from live_signal_engine import Config, AggBar, Quote

# Load API key from .env
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('POLYGON_API_KEY='):
            api_key = line.split('=')[1].strip()
            break


class DataSourceTest:
    """Test all data sources"""
    
    def __init__(self):
        self.ws_agg_count = 0
        self.ws_quote_count = 0
        self.stop_flag = False
    
    def on_agg_callback(self, agg: AggBar) -> None:
        """Called when WebSocket aggregate arrives"""
        self.ws_agg_count += 1
        logger.info(
            f"[WS AGG #{self.ws_agg_count}] {agg.symbol} @ {agg.end_ts.strftime('%H:%M:%S')} "
            f"O={agg.o:.2f} H={agg.h:.2f} L={agg.l:.2f} C={agg.c:.2f} V={agg.v:.0f}"
        )
        
        if self.ws_agg_count >= 10:
            self.stop_flag = True
    
    def on_quote_callback(self, quote: Quote) -> None:
        """Called when WebSocket quote arrives"""
        self.ws_quote_count += 1
        spread = quote.ask - quote.bid
        logger.info(
            f"[WS QUOTE #{self.ws_quote_count}] {quote.symbol} @ {quote.ts.strftime('%H:%M:%S')} "
            f"Bid={quote.bid:.4f} Ask={quote.ask:.4f} Spread={spread:.4f}"
        )
    
    async def test_websocket(self) -> None:
        """Test WebSocket data streaming"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: WebSocket Real-Time Data")
        logger.info("=" * 80)
        logger.info("Connecting to Polygon WebSocket...")
        logger.info("Symbol: XAUUSD")
        logger.info("Channels: A.XAUUSD (aggregates), Q.XAUUSD (quotes)")
        logger.info("")
        
        try:
            client = PolygonWebSocketClient(
                api_key=api_key,
                symbol="XAUUSD",
                on_agg_callback=self.on_agg_callback,
                on_quote_callback=self.on_quote_callback,
            )
            
            # Run for up to 20 seconds or until we get 10 aggregates
            await asyncio.wait_for(client.connect(), timeout=20)
        
        except asyncio.TimeoutError:
            logger.info("✓ WebSocket test completed (timeout)")
        except KeyboardInterrupt:
            logger.info("✓ WebSocket test interrupted")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
        
        logger.info(f"\n✓ WebSocket received {self.ws_agg_count} aggregates, {self.ws_quote_count} quotes")
    
    def test_rest_api(self) -> None:
        """Test REST API data retrieval"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: REST API Data Retrieval")
        logger.info("=" * 80)
        
        client = PolygonRESTClient(api_key=api_key, symbol="XAUUSD")
        
        # Test 1: Current quote
        logger.info("\n[REST 1] Fetching current quote...")
        quote = client.get_current_quote()
        if quote:
            spread = quote.ask - quote.bid
            logger.info(
                f"✓ Current Quote: {quote.symbol} "
                f"Bid={quote.bid:.4f} Ask={quote.ask:.4f} Spread={spread:.4f} "
                f"@ {quote.ts.strftime('%H:%M:%S')}"
            )
        else:
            logger.warning("Could not fetch current quote")
        
        # Test 2: Previous bar
        logger.info("\n[REST 2] Fetching previous bar...")
        bar = client.get_previous_bar()
        if bar:
            logger.info(
                f"✓ Previous Bar: {bar.symbol} "
                f"OHLCV={bar.o:.2f}/{bar.h:.2f}/{bar.l:.2f}/{bar.c:.2f}/{bar.v:.0f} "
                f"@ {bar.end_ts.strftime('%Y-%m-%d')}"
            )
        else:
            logger.warning("Could not fetch previous bar")
        
        # Test 3: Today's bar
        logger.info("\n[REST 3] Fetching today's bar...")
        today_bar = client.get_todays_bar()
        if today_bar:
            logger.info(
                f"✓ Today's Bar: {today_bar.symbol} "
                f"OHLCV={today_bar.o:.2f}/{today_bar.h:.2f}/{today_bar.l:.2f}/{today_bar.c:.2f}/{today_bar.v:.0f} "
                f"@ {today_bar.end_ts.strftime('%Y-%m-%d')}"
            )
        else:
            logger.info("Today's bar not available (market may be closed)")
    
    async def run(self) -> None:
        """Run all tests"""
        logger.info("\n" + "=" * 80)
        logger.info("POLYGON DATA SOURCE COMPREHENSIVE TEST")
        logger.info("=" * 80)
        logger.info(f"API Key: {api_key[:10]}...")
        logger.info(f"Premium Tier: Yes")
        logger.info("")
        
        # Test WebSocket
        await self.test_websocket()
        
        # Test REST API
        self.test_rest_api()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"WebSocket Aggregates: {self.ws_agg_count}")
        logger.info(f"WebSocket Quotes: {self.ws_quote_count}")
        logger.info(f"REST API: Tested 3 endpoints")
        logger.info("=" * 80)


if __name__ == "__main__":
    test = DataSourceTest()
    asyncio.run(test.run())
