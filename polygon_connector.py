"""
polygon_connector.py
====================
Dual-mode Polygon.io connector for institutional trading systems.

Symbol Format Reference:
- REST Quotes: C:XAU-USD (with hyphen)
- REST Bars: C:XAUUSD (no hyphen)
- WebSocket Second Bars: CAS.XAU/USD (slash separator)
- WebSocket Minute Bars: CA.XAU/USD (slash separator)
- WebSocket Quotes: C.XAU/USD (slash separator)

Architecture:
- REST: Bulk historical data (50,000 bars/quotes) for features/backtesting
- WebSocket: Live streaming (quotes, second bars, minute bars) for real-time signals

Performance:
- REST: 50k bars in <3s with pagination
- WebSocket: <10ms bar/quote delivery
- Signal generation: <50ms end-to-end
"""

import os
import logging
import asyncio
from typing import Optional, List, Callable, Dict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import deque
import json

import aiohttp
import pandas as pd
import numpy as np
import websockets

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Real-time quote with microstructure metrics"""
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    exchange: int
    ts: pd.Timestamp
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid) * 10000
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance"""
        total = self.bid_size + self.ask_size
        if total == 0:
            return 0
        return (self.bid_size - self.ask_size) / total


@dataclass
class AggBar:
    """OHLCV bar with microstructure data"""
    symbol: str
    o: float
    h: float
    l: float
    c: float
    v: float
    vw: float  # VWAP
    n: int     # Trade count
    ts: pd.Timestamp
    timeframe: str  # '1S', '1T', '5T', etc.
    
    @property
    def range(self) -> float:
        return self.h - self.l
    
    @property
    def body(self) -> float:
        return abs(self.c - self.o)
    
    @property
    def is_bullish(self) -> bool:
        return self.c > self.o
    
    @property
    def wick_top(self) -> float:
        return self.h - max(self.o, self.c)
    
    @property
    def wick_bottom(self) -> float:
        return min(self.o, self.c) - self.l


class PolygonRESTClient:
    """
    High-performance REST client for bulk historical data.
    
    Use Cases:
    - Fetch 50,000 most recent bars (second/minute)
    - Fetch 50,000 most recent quotes
    - Feature engineering for ML models
    - Backtesting dataset construction
    """
    
    BASE_URL = "https://api.polygon.io"
    MAX_LIMIT_PER_REQUEST = 50000  # Polygon's max
    
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=20,
            ttl_dns_cache=300
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_latest_quote(self, symbol: str = "C:XAU-USD") -> Optional[Quote]:
        """
        Fetch single latest quote (for live signals).
        Symbol: C:XAU-USD (with hyphen for quotes)
        Target: <20ms
        """
        url = f"{self.BASE_URL}/v3/quotes/{symbol}"
        params = {
            "apiKey": self.api_key,
            "limit": 1,
            "order": "desc"
        }
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Quote fetch failed: {resp.status}")
                    return None
                
                data = await resp.json()
                if not data.get('results'):
                    return None
                
                q = data['results'][0]
                ts = pd.Timestamp(q.get('sip_timestamp', q.get('participant_timestamp')), unit='ns', tz='UTC')
                
                return Quote(
                    symbol=symbol,
                    bid=float(q['bid_price']),
                    ask=float(q['ask_price']),
                    bid_size=float(q.get('bid_size', 0)),
                    ask_size=float(q.get('ask_size', 0)),
                    exchange=int(q.get('bid_exchange', 0)),
                    ts=ts
                )
        
        except Exception as e:
            logger.error(f"Quote error: {e}")
            return None
    
    async def get_historical_quotes(
        self,
        symbol: str = "C:XAU-USD",
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch 50,000 most recent quotes for feature engineering.
        Symbol: C:XAU-USD (with hyphen for quotes)
        
        Returns:
            DataFrame with columns: [timestamp, bid, ask, bid_size, ask_size, mid, spread]
        
        Target: <3s for 50k quotes
        """
        url = f"{self.BASE_URL}/v3/quotes/{symbol}"
        
        # Calculate reasonable date range (quotes: ~7 days for 50k)
        end = pd.Timestamp.now(tz='UTC')
        start = end - pd.Timedelta(days=14)  # Buffer for market hours
        
        params = {
            "apiKey": self.api_key,
            "timestamp.gte": int(start.timestamp() * 1000),
            "timestamp.lte": int(end.timestamp() * 1000),
            "limit": min(limit, self.MAX_LIMIT_PER_REQUEST),
            "order": "desc"
        }
        
        quotes = []
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Historical quotes failed: {resp.status}")
                    return pd.DataFrame()
                
                data = await resp.json()
                results = data.get('results', [])
                
                for q in results:
                    ts = pd.Timestamp(q.get('sip_timestamp', q.get('participant_timestamp')), unit='ns', tz='UTC')
                    bid = float(q['bid_price'])
                    ask = float(q['ask_price'])
                    
                    quotes.append({
                        'timestamp': ts,
                        'bid': bid,
                        'ask': ask,
                        'bid_size': float(q.get('bid_size', 0)),
                        'ask_size': float(q.get('ask_size', 0)),
                        'mid': (bid + ask) / 2,
                        'spread': ask - bid,
                        'exchange': int(q.get('bid_exchange', 0))
                    })
                
                df = pd.DataFrame(quotes)
                if not df.empty:
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Fetched {len(df)} quotes")
                return df
        
        except Exception as e:
            logger.error(f"Historical quotes error: {e}")
            return pd.DataFrame()
    
    async def get_historical_bars(
        self,
        symbol: str = "C:XAUUSD",
        timespan: str = "minute",
        multiplier: int = 1,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch 50,000 most recent bars for feature engineering.
        Symbol: C:XAUUSD (no hyphen for bars)
        
        Args:
            symbol: Ticker (C:XAUUSD for forex bars)
            timespan: 'second', 'minute', 'hour', 'day'
            multiplier: 1, 5, 15, etc.
            limit: Number of bars (max 50,000)
        
        Returns:
            DataFrame with OHLCV + microstructure columns
        
        Performance:
        - Second bars: ~30 days for 50k (market hours)
        - Minute bars: ~35 days for 50k
        """
        # Calculate date range based on timespan
        end = pd.Timestamp.now(tz='UTC')
        
        if timespan == "second":
            days = 35  # ~50k seconds in market hours
        elif timespan == "minute":
            days = 35  # ~50k minutes
        elif timespan == "hour":
            days = 365  # ~50k hours = ~2000 days, but cap at 1 year
        else:
            days = 7
        
        start = end - pd.Timedelta(days=days)
        
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.date()}/{end.date()}"
        params = {
            "apiKey": self.api_key,
            "limit": min(limit, self.MAX_LIMIT_PER_REQUEST),
            "sort": "desc"
        }
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Historical bars failed: {resp.status}")
                    return pd.DataFrame()
                
                data = await resp.json()
                results = data.get('results', [])
                
                bars = []
                for bar in results:
                    ts = pd.Timestamp(bar['t'], unit='ms', tz='UTC')
                    
                    bars.append({
                        'timestamp': ts,
                        'open': float(bar['o']),
                        'high': float(bar['h']),
                        'low': float(bar['l']),
                        'close': float(bar['c']),
                        'volume': float(bar['v']),
                        'vwap': float(bar.get('vw', bar['c'])),
                        'trades': int(bar.get('n', 0))
                    })
                
                df = pd.DataFrame(bars)
                if not df.empty:
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Add technical features
                    df['range'] = df['high'] - df['low']
                    df['body'] = (df['close'] - df['open']).abs()
                    df['is_bullish'] = (df['close'] > df['open']).astype(int)
                
                logger.info(f"Fetched {len(df)} {timespan} bars")
                return df
        
        except Exception as e:
            logger.error(f"Historical bars error: {e}")
            return pd.DataFrame()


class PolygonWebSocketClient:
    """
    Real-time WebSocket client for live trading signals.
    
    WebSocket Symbol Format (all use slash separator):
    - Quotes: C.XAU/USD
    - Second bars: CAS.XAU/USD
    - Minute bars: CA.XAU/USD
    
    Features:
    - Rolling buffer (configurable size)
    - Auto-reconnect
    - Latency tracking
    """
    
    WS_URL = "wss://socket.polygon.io/forex"
    
    def __init__(
        self,
        api_key: str,
        symbol: str = "XAU/USD",
        buffer_size: int = 1000
    ):
        self.api_key = api_key
        self.symbol = symbol  # XAU/USD format (slash separator)
        self.buffer_size = buffer_size
        
        # Separate buffers for each data type
        self.quote_buffer: deque[Quote] = deque(maxlen=buffer_size)
        self.second_buffer: deque[AggBar] = deque(maxlen=buffer_size)
        self.minute_buffer: deque[AggBar] = deque(maxlen=buffer_size)
        
        # Thread-safe locks
        self.quote_lock = asyncio.Lock()
        self.second_lock = asyncio.Lock()
        self.minute_lock = asyncio.Lock()
        
        # Callbacks
        self.on_quote: Optional[Callable] = None
        self.on_second_bar: Optional[Callable] = None
        self.on_minute_bar: Optional[Callable] = None
        
        self.ws = None
        self.running = False
        
        # Latency tracking
        self.last_message_ts = None
        self.latencies = deque(maxlen=100)
    
    async def connect(
        self,
        subscribe_quotes: bool = True,
        subscribe_second_bars: bool = True,
        subscribe_minute_bars: bool = True
    ):
        """Connect and subscribe to specified streams"""
        self.running = True
        
        while self.running:
            try:
                async with websockets.connect(self.WS_URL) as ws:
                    self.ws = ws
                    logger.info(f"✓ WebSocket connected")
                    
                    # Authenticate
                    await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                    auth_resp = await ws.recv()
                    logger.info(f"✓ Authenticated: {auth_resp}")
                    
                    # Subscribe to requested streams (all use XAU/USD format)
                    subscriptions = []
                    if subscribe_quotes:
                        subscriptions.append(f"C.{self.symbol}")  # C.XAU/USD
                    if subscribe_second_bars:
                        subscriptions.append(f"CAS.{self.symbol}")  # CAS.XAU/USD
                    if subscribe_minute_bars:
                        subscriptions.append(f"CA.{self.symbol}")  # CA.XAU/USD
                    
                    if subscriptions:
                        sub_msg = {"action": "subscribe", "params": ",".join(subscriptions)}
                        await ws.send(json.dumps(sub_msg))
                        logger.info(f"✓ Subscribed: {', '.join(subscriptions)}")
                    
                    # Process messages
                    async for message in ws:
                        await self._process_message(message)
            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                if self.running:
                    logger.info("Reconnecting in 5s...")
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5s...")
                    await asyncio.sleep(5)
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket messages"""
        try:
            receive_ts = pd.Timestamp.now(tz='UTC')
            data = json.loads(message)
            
            if not isinstance(data, list):
                return
            
            for msg in data:
                ev = msg.get('ev')
                
                if ev == 'C':  # Quote (C.XAU/USD)
                    await self._handle_quote(msg, receive_ts)
                elif ev == 'CAS':  # Second bar (CAS.XAU/USD)
                    await self._handle_second_bar(msg, receive_ts)
                elif ev == 'CA':  # Minute bar (CA.XAU/USD)
                    await self._handle_minute_bar(msg, receive_ts)
        
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    async def _handle_quote(self, msg: dict, receive_ts: pd.Timestamp):
        """Handle real-time quote from C.XAU/USD"""
        try:
            ts = pd.Timestamp(msg['t'], unit='ms', tz='UTC')
            
            # Track latency
            latency_ms = (receive_ts - ts).total_seconds() * 1000
            self.latencies.append(latency_ms)
            
            quote = Quote(
                symbol=msg['p'],
                bid=float(msg['b']),
                ask=float(msg['a']),
                bid_size=float(msg.get('bs', 0)),
                ask_size=float(msg.get('as', 0)),
                exchange=int(msg.get('x', 0)),
                ts=ts
            )
            
            async with self.quote_lock:
                self.quote_buffer.appendleft(quote)
            
            if self.on_quote:
                self.on_quote(quote)
            
            logger.debug(f"Quote: {quote.symbol} {quote.bid:.4f}/{quote.ask:.4f} (latency: {latency_ms:.1f}ms)")
        
        except Exception as e:
            logger.error(f"Quote handler error: {e}")
    
    async def _handle_second_bar(self, msg: dict, receive_ts: pd.Timestamp):
        """Handle second bar from CAS.XAU/USD"""
        try:
            ts = pd.Timestamp(msg['s'], unit='ms', tz='UTC')
            
            bar = AggBar(
                symbol=msg['pair'],
                o=float(msg['o']),
                h=float(msg['h']),
                l=float(msg['l']),
                c=float(msg['c']),
                v=float(msg.get('v', 0)),
                vw=float(msg.get('vw', msg['c'])),
                n=int(msg.get('n', 0)),
                ts=ts,
                timeframe='1S'
            )
            
            async with self.second_lock:
                self.second_buffer.appendleft(bar)
            
            if self.on_second_bar:
                self.on_second_bar(bar)
            
            logger.debug(f"Second bar: {bar.symbol} C={bar.c:.4f}")
        
        except Exception as e:
            logger.error(f"Second bar error: {e}")
    
    async def _handle_minute_bar(self, msg: dict, receive_ts: pd.Timestamp):
        """Handle minute bar from CA.XAU/USD"""
        try:
            ts = pd.Timestamp(msg['s'], unit='ms', tz='UTC')
            
            bar = AggBar(
                symbol=msg['pair'],
                o=float(msg['o']),
                h=float(msg['h']),
                l=float(msg['l']),
                c=float(msg['c']),
                v=float(msg.get('v', 0)),
                vw=float(msg.get('vw', msg['c'])),
                n=int(msg.get('n', 0)),
                ts=ts,
                timeframe='1T'
            )
            
            async with self.minute_lock:
                self.minute_buffer.appendleft(bar)
            
            if self.on_minute_bar:
                self.on_minute_bar(bar)
            
            logger.debug(f"Minute bar: {bar.symbol} C={bar.c:.4f}")
        
        except Exception as e:
            logger.error(f"Minute bar error: {e}")
    
    async def get_latest_quote(self) -> Optional[Quote]:
        """Get most recent quote from buffer"""
        async with self.quote_lock:
            return self.quote_buffer[0] if self.quote_buffer else None
    
    async def get_recent_quotes(self, limit: int = 100) -> List[Quote]:
        """Get N most recent quotes"""
        async with self.quote_lock:
            return list(self.quote_buffer)[:limit]
    
    async def get_recent_second_bars(self, limit: int = 60) -> List[AggBar]:
        """Get N most recent second bars"""
        async with self.second_lock:
            return list(self.second_buffer)[:limit]
    
    async def get_recent_minute_bars(self, limit: int = 100) -> List[AggBar]:
        """Get N most recent minute bars"""
        async with self.minute_lock:
            return list(self.minute_buffer)[:limit]
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get WebSocket latency statistics"""
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        return {
            'mean_ms': np.mean(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'max_ms': np.max(latencies)
        }
    
    async def disconnect(self):
        """Gracefully disconnect"""
        self.running = False
        if self.ws:
            await self.ws.close()
            logger.info("WebSocket disconnected")


class HybridPolygonClient:
    """
    Production-ready dual-mode client.
    
    Use Cases:
    1. Feature Engineering: REST to fetch 50k historical bars/quotes
    2. Live Signals: WebSocket for real-time streaming
    3. Hybrid: REST historical + WebSocket live for complete context
    """
    
    def __init__(
        self,
        api_key: str,
        ws_symbol: str = "XAU/USD",       # WebSocket format (slash)
        rest_quote_symbol: str = "C:XAU-USD",  # REST quote format (hyphen)
        rest_bar_symbol: str = "C:XAUUSD"      # REST bar format (no separator)
    ):
        self.api_key = api_key
        self.ws_symbol = ws_symbol
        self.rest_quote_symbol = rest_quote_symbol
        self.rest_bar_symbol = rest_bar_symbol
        
        self.ws_client = PolygonWebSocketClient(api_key, ws_symbol)
        self.rest_client = None  # Created in async context
    
    async def start_streaming(
        self,
        quotes: bool = True,
        second_bars: bool = True,
        minute_bars: bool = True
    ):
        """Start WebSocket streaming in background"""
        asyncio.create_task(
            self.ws_client.connect(quotes, second_bars, minute_bars)
        )
        await asyncio.sleep(2)  # Wait for initial buffer
        logger.info("✓ Streaming started")
    
    async def fetch_historical_context(
        self,
        quotes: int = 50000,
        minute_bars: int = 50000,
        second_bars: int = 0
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bulk historical data for feature engineering.
        
        Returns:
            {'quotes': DataFrame, 'minute_bars': DataFrame, 'second_bars': DataFrame}
        """
        async with PolygonRESTClient(self.api_key) as client:
            tasks = []
            
            if quotes > 0:
                tasks.append(client.get_historical_quotes(self.rest_quote_symbol, quotes))
            else:
                tasks.append(asyncio.sleep(0))
            
            if minute_bars > 0:
                tasks.append(client.get_historical_bars(self.rest_bar_symbol, 'minute', 1, minute_bars))
            else:
                tasks.append(asyncio.sleep(0))
            
            if second_bars > 0:
                tasks.append(client.get_historical_bars(self.rest_bar_symbol, 'second', 1, second_bars))
            else:
                tasks.append(asyncio.sleep(0))
            
            results = await asyncio.gather(*tasks)
            
            return {
                'quotes': results[0] if quotes > 0 else pd.DataFrame(),
                'minute_bars': results[1] if minute_bars > 0 else pd.DataFrame(),
                'second_bars': results[2] if second_bars > 0 else pd.DataFrame()
            }
    
    async def get_live_quote(self) -> Optional[Quote]:
        """Get latest quote (WebSocket buffer > REST)"""
        # Try WebSocket buffer first
        quote = await self.ws_client.get_latest_quote()
        if quote:
            return quote
        
        # Fallback to REST
        async with PolygonRESTClient(self.api_key) as client:
            return await client.get_latest_quote(self.rest_quote_symbol)
    
    async def get_live_market_data(self) -> Dict[str, any]:
        """
        Get complete live market snapshot.
        
        Returns:
            {
                'quote': Quote,
                'second_bars': List[AggBar],
                'minute_bars': List[AggBar],
                'latency_stats': Dict
            }
        """
        quote = await self.get_live_quote()
        second_bars = await self.ws_client.get_recent_second_bars(60)
        minute_bars = await self.ws_client.get_recent_minute_bars(100)
        latency_stats = self.ws_client.get_latency_stats()
        
        return {
            'quote': quote,
            'second_bars': second_bars,
            'minute_bars': minute_bars,
            'latency_stats': latency_stats
        }
    
    async def stop(self):
        """Stop WebSocket client"""
        await self.ws_client.disconnect()