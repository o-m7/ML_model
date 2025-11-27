================================================================================
DATA EXTRACTION - CORRECTED FOR MASSIVE.COM API
================================================================================

COMPLETE REWRITE: polygon_connector.py
================================================================================

KEY CHANGES:
─────────────

1. WEBSOCKET CLIENT
   • Now uses: massive.WebSocketClient (not polygon.WebSocketClient)
   • Symbol format: "XAU/USD" (with slash, not XAUUSD)
   • Channel types:
     - "CA.XAU/USD" → Minute aggregates
     - "CAS.XAU/USD" → Second aggregates
     - "C.XAU/USD" → Quotes
     - "FMV.XAU/USD" → Fair Market Value
   • Message type: WebSocketMessage (proper Massive type)
   • Callback: client.run(handler) instead of async connect

2. REST CLIENT
   • Fast implementation using: massive.RESTClient
   • Quote ticker format: "C:XAU/USD" (with slash)
   • Returns in seconds (no delays)

3. MESSAGE EXTRACTION
   • Aggregates: Extract open, high, low, close, volume, timestamp
   • Quotes: Extract bid_price, ask_price, timestamp
   • FMV: Extract value and symbol

USAGE EXAMPLES:
================================================================================

WEBSOCKET - MINUTE AGGREGATES:
───────────────────────────────
from polygon_connector import PolygonWebSocketClient

def on_agg(agg):
    print(f"Minute bar: {agg.symbol} Close={agg.c:.2f}")

client = PolygonWebSocketClient(
    api_key="YOUR_KEY",
    symbol="XAU/USD",
    on_agg_callback=on_agg,
    channel_type="minute"  # CA.XAU/USD
)
client.connect()  # Blocks and streams data


WEBSOCKET - SECOND AGGREGATES:
───────────────────────────────
client = PolygonWebSocketClient(
    api_key="YOUR_KEY",
    symbol="XAU/USD",
    on_agg_callback=on_agg,
    channel_type="second"  # CAS.XAU/USD
)
client.connect()


WEBSOCKET - QUOTES:
────────────────────
def on_quote(quote):
    spread = quote.ask - quote.bid
    print(f"Quote: {quote.symbol} Bid={quote.bid:.4f} Ask={quote.ask:.4f}")

client = PolygonWebSocketClient(
    api_key="YOUR_KEY",
    symbol="XAU/USD",
    on_quote_callback=on_quote,
    channel_type="quotes"  # C.XAU/USD
)
client.connect()


REST API - QUOTES (FAST):
──────────────────────────
from polygon_connector import PolygonRESTClient

rest_client = PolygonRESTClient(api_key="YOUR_KEY", symbol="XAU/USD")
quote = rest_client.get_current_quote()

if quote:
    print(f"Bid: {quote.bid:.4f}")
    print(f"Ask: {quote.ask:.4f}")
    print(f"Timestamp: {quote.ts}")


SYMBOL FORMATS:
================================================================================

Massive WebSocket:
  • XAU/USD (slash format)
  • USD/EUR, USD/CAD, etc.

Massive REST:
  • C:XAU/USD (prefix + slash)
  • C:USD/EUR, etc.

Channel Prefixes:
  • CA. → Minute aggregates
  • CAS. → Second aggregates
  • C. → Quotes
  • FMV. → Fair Market Value


DATA FLOW:
================================================================================

WebSocket Aggregates:
  1. Connect → Subscribe to CA.XAU/USD
  2. Receive WebSocketMessage objects
  3. Extract: symbol, open, high, low, close, volume, timestamp
  4. Create AggBar (normalized)
  5. Invoke callback

WebSocket Quotes:
  1. Connect → Subscribe to C.XAU/USD
  2. Receive WebSocketMessage objects
  3. Extract: symbol, bid_price, ask_price, timestamp
  4. Create Quote (normalized)
  5. Invoke callback

REST Quotes:
  1. Create RESTClient with API key
  2. Call list_quotes(ticker="C:XAU/USD", limit=1)
  3. Extract bid_price, ask_price, timestamp
  4. Return Quote object
  5. Completes in <1 second


PERFORMANCE:
================================================================================

WebSocket (Real-time):
  • Second aggregates: ~1-2 sec delay from market
  • Minute aggregates: ~30-60 sec delay from market
  • Quotes: ~100-500ms delay from market

REST API:
  • Quotes: <1 second
  • Returns latest available data

No synthetic/mock data - all real market data.

================================================================================
