#!/usr/bin/env python3
"""
Test WebSocket message extraction from Polygon.io
Validates CurrencyAgg and ForexQuote message handling
"""

import asyncio
import logging
import os
from typing import List
from datetime import datetime, timezone
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Load credentials
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

if not POLYGON_API_KEY:
    print("❌ POLYGON_API_KEY not found in environment")
    exit(1)

# Import after env is loaded
from polygon import WebSocketClient
from polygon.websocket.models.common import Feed, Market
from polygon.websocket.models import CurrencyAgg, ForexQuote


class WebSocketMessageTest:
    """Test and validate WebSocket message extraction"""
    
    def __init__(self):
        self.agg_count = 0
        self.quote_count = 0
        self.test_duration = 30  # seconds
        self.start_time = None
    
    async def process_messages(self, messages: List) -> None:
        """Process WebSocket messages and extract data"""
        try:
            if not isinstance(messages, list):
                messages = [messages]
            
            for msg in messages:
                # Check message type and extract data
                if isinstance(msg, CurrencyAgg):
                    self._handle_agg(msg)
                elif isinstance(msg, ForexQuote):
                    self._handle_quote(msg)
                
                # Stop after test duration
                if self.agg_count + self.quote_count >= 50:
                    raise KeyboardInterrupt("Test complete: 50+ messages received")
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Error processing: {e}", exc_info=True)
    
    def _handle_agg(self, msg: CurrencyAgg) -> None:
        """Extract and log aggregate data"""
        try:
            self.agg_count += 1
            
            symbol = msg.pair
            o = msg.open
            h = msg.high
            l = msg.low
            c = msg.close
            v = msg.volume
            start_ts = msg.start_timestamp
            end_ts = msg.end_timestamp
            vwap = msg.vwap
            
            # Convert timestamps
            end_dt = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)
            
            logger.info(
                f"[{self.agg_count:03d}] AGG: {symbol} @ {end_dt.strftime('%H:%M:%S')} "
                f"O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} V={v:.0f} VWAP={vwap:.2f}"
            )
        
        except Exception as e:
            logger.error(f"Aggregate error: {e}", exc_info=True)
    
    def _handle_quote(self, msg: ForexQuote) -> None:
        """Extract and log quote data"""
        try:
            self.quote_count += 1
            
            symbol = msg.pair
            bid = msg.bid_price
            ask = msg.ask_price
            ts = msg.timestamp
            spread = ask - bid
            
            # Convert timestamp
            ts_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            
            logger.info(
                f"[{self.quote_count:03d}] QUOTE: {symbol} @ {ts_dt.strftime('%H:%M:%S')} "
                f"Bid={bid:.4f} Ask={ask:.4f} Spread={spread:.4f}"
            )
        
        except Exception as e:
            logger.error(f"Quote error: {e}", exc_info=True)
    
    async def run(self) -> None:
        """Connect and receive messages"""
        logger.info("=" * 80)
        logger.info("POLYGON WebSocket Message Test")
        logger.info("=" * 80)
        logger.info(f"Symbol: XAUUSD")
        logger.info(f"Feed: RealTime (socket.polygon.io)")
        logger.info(f"Market: Forex")
        logger.info(f"Subscribing to: A.XAUUSD (aggregates), Q.XAUUSD (quotes)")
        logger.info("")
        
        try:
            # Create WebSocket client
            client = WebSocketClient(
                api_key=POLYGON_API_KEY,
                feed=Feed.RealTime,
                market=Market.Forex,
                verbose=False,
            )
            
            logger.info("✓ WebSocket client created")
            
            # Subscribe to both aggregate and quote channels
            client.subscribe("A.XAUUSD", "Q.XAUUSD")
            logger.info("✓ Subscribed to A.XAUUSD and Q.XAUUSD")
            logger.info("")
            
            # Connect with message processor
            self.start_time = datetime.now(timezone.utc)
            logger.info("🔌 Connecting to socket.polygon.io...")
            logger.info("")
            
            await asyncio.wait_for(
                client.connect(processor=self.process_messages),
                timeout=self.test_duration
            )
        
        except asyncio.TimeoutError:
            logger.info(f"\n✓ Test completed (timeout after {self.test_duration}s)")
        
        except KeyboardInterrupt as e:
            logger.info(f"\n✓ Test completed: {e}")
        
        except Exception as e:
            logger.error(f"❌ Connection error: {e}", exc_info=True)
            raise
        
        finally:
            self._print_summary()
    
    def _print_summary(self) -> None:
        """Print test results"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Aggregates received: {self.agg_count}")
        logger.info(f"Quotes received: {self.quote_count}")
        logger.info(f"Total messages: {self.agg_count + self.quote_count}")
        logger.info("")
        
        if self.agg_count > 0:
            logger.info("✓ CurrencyAgg messages working correctly")
        else:
            logger.warning("⚠ No CurrencyAgg messages received")
        
        if self.quote_count > 0:
            logger.info("✓ ForexQuote messages working correctly")
        else:
            logger.warning("⚠ No ForexQuote messages received")
        
        logger.info("=" * 80)


if __name__ == "__main__":
    test = WebSocketMessageTest()
    asyncio.run(test.run())
