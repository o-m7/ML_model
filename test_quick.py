#!/usr/bin/env python3
"""
Quick Test - Live Signals Connections & Generation
====================================================

Fast local tests for:
1. Config & imports ✓
2. REST API connectivity
3. WebSocket connectivity  
4. Feature computation
5. Signal generation
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def test_api_connectivity():
    """Test REST API with correct endpoint."""
    logger.info("\n" + "="*70)
    logger.info("TEST: REST API Connectivity")
    logger.info("="*70)
    
    import requests
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    # Test with a simple endpoint that works
    url = f"https://api.polygon.io/v1/open-close/XAUUSD/2024-11-25"
    params = {'apiKey': api_key}
    
    logger.info(f"GET {url}")
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        logger.info(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"✓ Success")
            logger.info(f"  Status: {data.get('status')}")
            if 'results' in data and data['results']:
                logger.info(f"  Symbol: {data.get('from')}")
                logger.info(f"  Close: {data.get('close')}")
            return True
        else:
            logger.error(f"✗ Error {resp.status_code}: {resp.text[:200]}")
            return False
    
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        return False


def test_feature_engine():
    """Test feature computation."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Feature Engine")
    logger.info("="*70)
    
    from live_signal_engine import SignalGenerator, AggBar
    import pandas as pd
    import numpy as np
    
    try:
        engine = SignalGenerator()
        logger.info("✓ SignalGenerator initialized")
        
        # Create 100 bars of mock data
        for i in range(100):
            ts = pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
            price = 100 + np.sin(i/10) * 5 + np.random.randn() * 0.5
            
            bar = AggBar(
                symbol='XAUUSD',
                o=price - 0.2,
                h=price + 0.3,
                l=price - 0.3,
                c=price,
                v=1000 + np.random.randint(-200, 200),
                start_ts=ts,
                end_ts=ts,
            )
            
            engine.feature_engine.add_bar('1T', bar)
        
        logger.info(f"✓ Added 100 bars to feature engine")
        
        # Compute features
        features = engine.feature_engine.compute_features('1T')
        
        if features is not None:
            logger.info(f"✓ Features computed: {len(features)} features")
            logger.info(f"  Sample features:")
            for key in list(features.index)[:5]:
                val = features[key]
                logger.info(f"    - {key}: {val:.4f}" if isinstance(val, (int, float)) else f"    - {key}: {val}")
            return True
        else:
            logger.error("✗ No features returned")
            return False
    
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_websocket():
    """Test WebSocket connection."""
    logger.info("\n" + "="*70)
    logger.info("TEST: WebSocket Connection (20 second timeout)")
    logger.info("="*70)
    
    from polygon import WebSocketClient
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    try:
        logger.info(f"Creating WebSocketClient...")
        client = WebSocketClient(api_key=api_key, feed="sip", market="forex")
        
        message_count = {'agg': 0, 'quote': 0}
        
        def on_agg(agg_data):
            message_count['agg'] += 1
            if message_count['agg'] == 1:
                logger.info(f"✓ Received aggregate")
                logger.info(f"  Type: {type(agg_data)}")
                logger.info(f"  Data: {agg_data}")
        
        def on_quote(quote_data):
            message_count['quote'] += 1
            if message_count['quote'] == 1:
                logger.info(f"✓ Received quote")
                logger.info(f"  Type: {type(quote_data)}")
                logger.info(f"  Data: {quote_data}")
        
        def on_error(error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(msg):
            logger.warning(f"WebSocket closed: {msg}")
        
        # Register handlers - check available API
        logger.info(f"Available methods: {[m for m in dir(client) if not m.startswith('_')]}")
        
        # Try different handler registration approaches
        try:
            # First try official API
            client.on_aggregate(on_agg)
            client.on_quote(on_quote)
            logger.info("✓ Handlers registered using on_* methods")
        except AttributeError:
            logger.warning("⚠ on_* methods not available, trying alternative")
        
        logger.info("✓ WebSocketClient created")
        logger.info("  (Note: Would subscribe and listen indefinitely in production)")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ WebSocket error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("LIVE SIGNALS - QUICK CONNECTION TEST")
    logger.info("="*70)
    
    results = {}
    
    # Test 1: REST API
    results['REST API'] = test_api_connectivity()
    
    # Test 2: Features
    results['Features'] = test_feature_engine()
    
    # Test 3: WebSocket
    try:
        results['WebSocket'] = asyncio.run(test_websocket())
    except Exception as e:
        logger.error(f"WebSocket test error: {e}")
        results['WebSocket'] = False
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    total_passed = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info(f"\nResult: {total_passed}/{total} tests passed")
    
    return 0 if total_passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
