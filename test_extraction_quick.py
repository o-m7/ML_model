#!/usr/bin/env python3
"""
Quick test: Verify corrected data extraction
- WebSocket: CurrencyAgg and ForexQuote objects
- REST API: Massive client with C:XAU-USD
- REST API: Fallback snapshot with C:XAUUSD
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from polygon_connector import PolygonRESTClient
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s | %(message)s')

# Read API key
api_key = Path('.env').read_text().split('POLYGON_API_KEY=')[1].split('\n')[0].strip()

print("=" * 80)
print("QUICK DATA EXTRACTION TEST")
print("=" * 80)

rest_client = PolygonRESTClient(api_key=api_key, symbol="XAUUSD")

print("\nSymbol Formats:")
print(f"  Base:    {rest_client.symbol_base}")
print(f"  OHLCV:   {rest_client.symbol_ohlcv}")
print(f"  Massive: {rest_client.symbol_massive}")

print("\n" + "-" * 80)
print("TEST 1: Get Current Quote (Massive client + fallback)")
print("-" * 80)
quote = rest_client.get_current_quote()
if quote:
    spread = quote.ask - quote.bid
    print(f"\n✓ Quote successfully fetched!")
    print(f"  Symbol:    {quote.symbol}")
    print(f"  Bid:       {quote.bid:.4f}")
    print(f"  Ask:       {quote.ask:.4f}")
    print(f"  Spread:    {spread:.4f}")
    print(f"  Timestamp: {quote.ts}")
else:
    print("✗ Failed to fetch quote")

print("\n" + "-" * 80)
print("TEST 2: Get Previous Bar (OHLCV with C:XAUUSD)")
print("-" * 80)
bar = rest_client.get_previous_bar()
if bar:
    print(f"\n✓ Previous bar successfully fetched!")
    print(f"  Symbol: {bar.symbol}")
    print(f"  Date:   {bar.end_ts.strftime('%Y-%m-%d')}")
    print(f"  Open:   {bar.o:.2f}")
    print(f"  High:   {bar.h:.2f}")
    print(f"  Low:    {bar.l:.2f}")
    print(f"  Close:  {bar.c:.2f}")
    print(f"  Volume: {bar.v:.0f}")
else:
    print("✗ Failed to fetch previous bar")

print("\n" + "-" * 80)
print("TEST 3: Get Today's Bar")
print("-" * 80)
today_bar = rest_client.get_todays_bar()
if today_bar:
    print(f"\n✓ Today's bar successfully fetched!")
    print(f"  Symbol: {today_bar.symbol}")
    print(f"  Date:   {today_bar.end_ts.strftime('%Y-%m-%d')}")
    print(f"  OHLCV:  {today_bar.o:.2f} / {today_bar.h:.2f} / {today_bar.l:.2f} / {today_bar.c:.2f} / {today_bar.v:.0f}")
else:
    print("⚠ Today's bar not available (market may be closed)")

print("\n" + "=" * 80)
print("✓ Data extraction test complete!")
print("=" * 80)
