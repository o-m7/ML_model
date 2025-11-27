#!/usr/bin/env python3
"""
Live WebSocket Stream Test
==========================

Connect to Polygon WebSocket and stream XAUUSD data.
Press Ctrl+C to stop.
"""

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv
from polygon_connector import PolygonWebSocketClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

msg_count = {'agg': 0, 'quote': 0}


def on_agg(agg):
    """Process aggregates."""
    msg_count['agg'] += 1
    logger.info(f"📊 Agg #{msg_count['agg']:3d}: {agg.symbol} @ {agg.end_ts.strftime('%H:%M:%S')} | OHLCV: {agg.o:.2f}/{agg.h:.2f}/{agg.l:.2f}/{agg.c:.2f}/{agg.v:>7.0f}")


def on_quote(quote):
    """Process quotes."""
    msg_count['quote'] += 1
    spread = (quote.ask - quote.bid) / quote.bid * 10000
    logger.info(f"💹 Quote #{msg_count['quote']:3d}: {quote.symbol} @ {quote.ts.strftime('%H:%M:%S')} | Bid: {quote.bid:.4f} Ask: {quote.ask:.4f} Spread: {spread:.1f}pips")


async def main():
    """Run live stream."""
    logger.info("\n" + "="*80)
    logger.info("LIVE WEBSOCKET STREAM - XAUUSD")
    logger.info("="*80)
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    logger.info(f"API Key: {api_key[:20]}...")
    logger.info("Connecting to Polygon WebSocket (socket.polygon.io)...")
    
    client = PolygonWebSocketClient(
        api_key=api_key,
        symbol='XAUUSD',
        on_agg_callback=on_agg,
        on_quote_callback=on_quote,
    )
    
    try:
        logger.info("Starting WebSocket connection...")
        await client.connect()
    except KeyboardInterrupt:
        logger.info("\n\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    logger.info(f"\nStream stopped")
    logger.info(f"Total aggregates received: {msg_count['agg']}")
    logger.info(f"Total quotes received: {msg_count['quote']}")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        sys.exit(0)
