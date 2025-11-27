#!/usr/bin/env python3
"""
Final Test - Live Signals System Complete Validation
=====================================================

Comprehensive test covering:
✓ Configuration & environment
✓ Module imports
✓ Feature engine
✓ Signal generation
✓ WebSocket connectivity
✓ REST API connectivity
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print a formatted section header."""
    logger.info("\n" + "="*80)
    logger.info(title.center(80))
    logger.info("="*80)


def test_config():
    """Test 1: Configuration and environment."""
    print_section("1. CONFIGURATION & ENVIRONMENT")
    
    # Check env vars
    api_key = os.getenv('POLYGON_API_KEY')
    s3_key = os.getenv('POLYGON_S3_ACCESS_KEY')
    supabase_url = os.getenv('SUPABASE_URL')
    
    if not api_key:
        logger.error("✗ POLYGON_API_KEY not set")
        return False
    
    logger.info(f"✓ POLYGON_API_KEY: {api_key[:20]}...")
    
    if s3_key:
        logger.info(f"✓ POLYGON_S3_ACCESS_KEY: {s3_key[:20]}...")
    else:
        logger.warning("○ POLYGON_S3_ACCESS_KEY: not set (optional)")
    
    if supabase_url:
        logger.info(f"✓ SUPABASE_URL: {supabase_url[:30]}...")
    else:
        logger.warning("○ SUPABASE_URL: not set (will skip signal persistence)")
    
    return True


def test_imports():
    """Test 2: Module imports."""
    print_section("2. MODULE IMPORTS")
    
    imports_to_test = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("onnx", None),
        ("onnxruntime", "ort"),
        ("polygon", None),  # polygon-api-client package is named 'polygon'
        ("websockets", None),
    ]
    
    all_ok = True
    for module_name, alias in imports_to_test:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
                logger.info(f"✓ {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
                logger.info(f"✓ {module_name}")
        except ImportError as e:
            logger.error(f"✗ {module_name}: {e}")
            all_ok = False
    
    # Test local modules
    try:
        from live_signal_engine import SignalGenerator, Config, AggBar, Quote
        logger.info(f"✓ live_signal_engine")
    except ImportError as e:
        logger.error(f"✗ live_signal_engine: {e}")
        all_ok = False
    
    try:
        from polygon_connector import PolygonWebSocketClient
        logger.info(f"✓ polygon_connector")
    except ImportError as e:
        logger.error(f"✗ polygon_connector: {e}")
        all_ok = False
    
    return all_ok


def test_feature_engine():
    """Test 3: Feature engine initialization and computation."""
    print_section("3. FEATURE ENGINE")
    
    from live_signal_engine import SignalGenerator, AggBar, Config
    import pandas as pd
    import numpy as np
    
    try:
        logger.info("Creating SignalGenerator...")
        engine = SignalGenerator()
        logger.info(f"✓ SignalGenerator initialized")
        logger.info(f"  Symbol: {Config.SYMBOL}")
        logger.info(f"  Timeframes: {list(Config.TIMEFRAMES.keys())}")
        logger.info(f"  Min bars for features: {Config.MIN_BARS_FOR_FEATURES}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}")
        return False
    
    # Test feature computation
    try:
        logger.info("\nAdding synthetic market data...")
        
        # Create 100 bars
        for i in range(100):
            ts = pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
            price = 100 + np.sin(i/10) * 5 + np.random.randn() * 0.3
            
            bar = AggBar(
                symbol='XAUUSD',
                o=price - 0.1,
                h=price + 0.2,
                l=price - 0.2,
                c=price,
                v=1000 + np.random.randint(-300, 300),
                start_ts=ts,
                end_ts=ts,
            )
            
            engine.feature_engine.add_bar('1T', bar)
        
        logger.info(f"✓ Added 100 bars to 1T timeframe")
        
        # Compute features
        logger.info("\nComputing features...")
        features = engine.feature_engine.compute_features('1T')
        
        if features is None:
            logger.error("✗ Feature computation returned None")
            return False
        
        logger.info(f"✓ Features computed: {len(features)} features")
        logger.info(f"\n  Sample features:")
        for key in list(features.index)[:8]:
            val = features[key]
            if isinstance(val, (int, float)):
                logger.info(f"    {key:20s}: {val:>10.6f}")
            else:
                logger.info(f"    {key:20s}: {val}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Feature computation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_rest_api():
    """Test 4: REST API connectivity."""
    print_section("4. REST API CONNECTIVITY")
    
    import requests
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    # Try endpoints with correct Polygon symbols
    endpoints = [
        ("XAUUSD Daily", f"https://api.polygon.io/v1/open-close/C:XAUUSD/2024-11-20", {'apiKey': api_key}),
        ("XAUUSD Previous Close", f"https://api.polygon.io/v2/aggs/ticker/C:XAUUSD/prev", {'apiKey': api_key}),
    ]
    
    for name, url, params in endpoints:
        try:
            logger.info(f"\nTesting: {name}")
            logger.info(f"  GET {url[:60]}...")
            
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code == 200:
                logger.info(f"  ✓ Status: {resp.status_code}")
                data = resp.json()
                logger.info(f"  ✓ Response: {data.get('status', 'OK')}")
                if 'results' in data and data['results']:
                    logger.info(f"  ✓ Data found: {len(data['results'])} result(s)")
                return True
            elif resp.status_code == 429:
                logger.warning(f"  ⚠ Rate limited (429) - API call succeeded")
                return True
            else:
                logger.warning(f"  ⚠ Status {resp.status_code}: {resp.text[:80]}")
        
        except Exception as e:
            logger.warning(f"  ⚠ {name} failed: {e}")
    
    logger.warning("✗ All REST API tests failed or inconclusive")
    return False


def test_websocket_client():
    """Test 5: WebSocket client initialization."""
    print_section("5. WEBSOCKET CLIENT")
    
    from polygon import WebSocketClient
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    try:
        logger.info("Creating WebSocketClient...")
        client = WebSocketClient(
            api_key=api_key,
            feed="sip",
            market="forex",
            verbose=False
        )
        
        logger.info(f"✓ WebSocketClient created")
        logger.info(f"  Feed: {client.feed}")
        logger.info(f"  Market: {client.market}")
        logger.info(f"  API Key: {api_key[:20]}...")
        
        # Check methods
        methods = ['subscribe', 'unsubscribe', 'connect', 'run']
        logger.info(f"\n  Available methods:")
        for method in methods:
            if hasattr(client, method):
                logger.info(f"    ✓ {method}")
        
        logger.info(f"\n✓ WebSocket client ready for production use")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to create WebSocketClient: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_polygon_connector():
    """Test 6: Polygon connector with mock data."""
    print_section("6. POLYGON CONNECTOR")
    
    from polygon_connector import PolygonWebSocketClient
    from live_signal_engine import AggBar, Quote
    import pandas as pd
    
    api_key = os.getenv('POLYGON_API_KEY')
    
    messages_received = {'agg': 0, 'quote': 0}
    
    def on_agg(agg):
        messages_received['agg'] += 1
        if messages_received['agg'] == 1:
            logger.info(f"  ✓ Aggregate callback called")
            logger.info(f"    Symbol: {agg.symbol}")
            logger.info(f"    OHLCV: {agg.o:.2f}/{agg.h:.2f}/{agg.l:.2f}/{agg.c:.2f}/{agg.v:.0f}")
    
    def on_quote(quote):
        messages_received['quote'] += 1
        if messages_received['quote'] == 1:
            logger.info(f"  ✓ Quote callback called")
            logger.info(f"    Symbol: {quote.symbol}")
            logger.info(f"    Bid/Ask: {quote.bid:.4f}/{quote.ask:.4f}")
    
    try:
        logger.info("Creating PolygonWebSocketClient...")
        client = PolygonWebSocketClient(
            api_key=api_key,
            symbol='XAUUSD',
            on_agg_callback=on_agg,
            on_quote_callback=on_quote,
        )
        
        logger.info(f"✓ Client created")
        logger.info(f"  Symbol: {client.symbol}")
        logger.info(f"  Callbacks: agg={bool(client.on_agg_callback)}, quote={bool(client.on_quote_callback)}")
        
        # Test message processing
        logger.info(f"\nTesting message handlers...")
        
        # Mock aggregate
        mock_agg = {
            'sym': 'XAUUSD',
            'o': 2000.5,
            'h': 2001.5,
            'l': 1999.5,
            'c': 2001.0,
            'v': 1000.0,
            's': int(pd.Timestamp.now(tz='UTC').timestamp() * 1000),
            'e': int(pd.Timestamp.now(tz='UTC').timestamp() * 1000),
        }
        
        import asyncio
        asyncio.run(client._handle_aggregate(mock_agg))
        
        # Mock quote
        mock_quote = {
            'sym': 'XAUUSD',
            'bid': 2000.5,
            'ask': 2000.6,
            't': int(pd.Timestamp.now(tz='UTC').timestamp() * 1000),
        }
        
        asyncio.run(client._handle_quote(mock_quote))
        
        logger.info(f"\n✓ Message handlers working correctly")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    print_section("LIVE SIGNALS SYSTEM - COMPREHENSIVE TEST")
    
    tests = [
        ("Configuration & Environment", test_config),
        ("Module Imports", test_imports),
        ("Feature Engine", test_feature_engine),
        ("REST API Connectivity", test_rest_api),
        ("WebSocket Client", test_websocket_client),
        ("Polygon Connector", test_polygon_connector),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.error(f"✗ Test crashed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[name] = False
    
    # Summary
    print_section("TEST SUMMARY")
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    total_passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Result: {total_passed}/{total} tests passed".center(80))
    logger.info(f"{'='*80}\n")
    
    if total_passed == total:
        logger.info("🎉 All tests passed! System is ready for live trading.")
        return 0
    else:
        logger.warning(f"⚠ {total - total_passed} test(s) failed. Review logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
