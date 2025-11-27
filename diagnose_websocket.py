#!/usr/bin/env python3
"""
Diagnostic - WebSocket Connection Issues
=========================================

Troubleshoots and identifies WebSocket connection problems.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
import socket

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def test_dns():
    """Test DNS resolution."""
    logger.info("\n" + "="*80)
    logger.info("1. DNS RESOLUTION TEST")
    logger.info("="*80)
    
    hosts = [
        'socket.polygon.io',
        'api.polygon.io',
        'files.polygon.io',
        'google.com',
    ]
    
    for host in hosts:
        try:
            ip = socket.gethostbyname(host)
            logger.info(f"✓ {host:30s} → {ip}")
        except socket.gaierror as e:
            logger.error(f"✗ {host:30s} → {e}")


def test_socket_connection():
    """Test TCP socket connection."""
    logger.info("\n" + "="*80)
    logger.info("2. TCP SOCKET CONNECTION TEST")
    logger.info("="*80)
    
    try:
        logger.info("Connecting to socket.polygon.io:443...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('socket.polygon.io', 443))
        logger.info("✓ Connected successfully")
        sock.close()
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")


async def test_websocket_connection():
    """Test WebSocket connection."""
    logger.info("\n" + "="*80)
    logger.info("3. WEBSOCKET CONNECTION TEST")
    logger.info("="*80)
    
    from polygon import WebSocketClient
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    try:
        logger.info("Creating WebSocketClient...")
        # Import Feed enum for correct feed specification
        from polygon.websocket.models.common import Feed, Market
        
        client = WebSocketClient(
            api_key=api_key,
            feed=Feed.RealTime,  # socket.polygon.io
            market=Market.Forex,  # Forex market
            verbose=True,
            max_reconnects=2,
        )
        
        logger.info("✓ Client created")
        logger.info(f"  Feed: {client.feed}")
        logger.info(f"  Market: {client.market}")
        
        # Subscribe
        client.subscribe("A.XAUUSD", "Q.XAUUSD")
        logger.info("✓ Subscribed to XAUUSD")
        
        # Create async message processor
        async def process_msg(msg):
            logger.info(f"Received message: {type(msg)}")
            if isinstance(msg, list):
                for m in msg:
                    logger.info(f"  - {m}")
            else:
                logger.info(f"  - {msg}")
            
            # Stop after first message for testing
            await client.close()
        
        logger.info("Attempting to connect...")
        try:
            await asyncio.wait_for(
                client.connect(processor=process_msg),
                timeout=10
            )
        except asyncio.TimeoutError:
            logger.warning("Connection timed out (may indicate successful connection but no data)")
            await client.close()
        
        logger.info("✓ WebSocket test completed")
        return True
    
    except Exception as e:
        logger.error(f"✗ WebSocket error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all diagnostics."""
    logger.info("\n" + "="*80)
    logger.info("WEBSOCKET DIAGNOSTICS")
    logger.info("="*80)
    
    # Test DNS
    test_dns()
    
    # Test socket
    test_socket_connection()
    
    # Test WebSocket
    try:
        result = asyncio.run(test_websocket_connection())
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        result = False
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("="*80)
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
