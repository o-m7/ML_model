"""
Polygon WebSocket tick collector using websocket-client.
Persists real ticks to local parquet files.
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger("polygon_ws_collector")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

DEFAULT_OUTPUT = Path("temp_download/ticks")
DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)



class PolygonWSCollector:
    def __init__(self, api_key: str = None, symbol: str = "C:XAUUSD", out_dir: Path = DEFAULT_OUTPUT):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("❌ POLYGON_API_KEY not set in environment")
        
        self.symbol = symbol
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.last_flush = time.time()
        self.ws = None
        self.tick_count = 0

    def _persist(self):
        """Persist buffered ticks to parquet."""
        if not self.buffer:
            return
        
        try:
            df = pd.DataFrame(self.buffer)
            
            # Normalize timestamp
            if 't' in df.columns and df['t'].notna().any():
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
            elif 'sip_timestamp' in df.columns and df['sip_timestamp'].notna().any():
                df['timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns', utc=True)
            else:
                df['timestamp'] = datetime.utcnow()
            
            # Get day for filename
            day = datetime.utcnow().strftime("%Y-%m-%d")
            out_file = self.out_dir / f"{self.symbol.replace(':', '_')}_{day}.parquet"
            
            # Append to existing file if it exists
            if out_file.exists():
                try:
                    existing = pd.read_parquet(out_file)
                    df = pd.concat([existing, df], ignore_index=True)
                    # Keep unique ticks (latest timestamp wins)
                    df = df.drop_duplicates(subset=['t', 'sip_timestamp'], keep='last', errors='ignore')
                except Exception as e:
                    logger.warning(f"Could not merge with existing parquet: {e}")
            
            # Write to disk
            df.to_parquet(out_file, index=False, compression='snappy')
            logger.info(f"✅ Persisted {len(self.buffer)} ticks to {out_file.name}")
            
            self.buffer = []
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error(f"❌ Persist error: {e}")

    def run(self, url: str = None):
        """Connect to Polygon WebSocket and stream ticks."""
        url = url or 'wss://socket.polygon.io/stocks'
        
        reconnect_attempts = 0
        max_reconnects = 20
        
        while reconnect_attempts < max_reconnects:
            try:
                from websocket._core import create_connection
                
                logger.info(f"🔗 Connecting to {url}")
                
                # Create connection with ping/pong
                self.ws = create_connection(url, ping_interval=30, ping_timeout=10)
                logger.info(f"✅ Connected")
                
                # Send auth
                auth_msg = {"action": "auth", "params": self.api_key}
                self.ws.send(json.dumps(auth_msg))
                logger.info(f"🔐 Sent auth")
                
                # Read auth response
                try:
                    auth_resp = self.ws.recv()
                    logger.info(f"📨 Auth response received")
                except Exception as e:
                    logger.warning(f"Could not read auth response: {e}")
                
                # Subscribe to quotes and trades
                subs = [f"Q.{self.symbol}", f"T.{self.symbol}"]
                for sub in subs:
                    sub_msg = {"action": "subscribe", "params": sub}
                    self.ws.send(json.dumps(sub_msg))
                    logger.info(f"📡 Subscribed to {sub}")
                
                # Listen for ticks
                reconnect_attempts = 0  # Reset on successful connection
                self._listen()
                
            except KeyboardInterrupt:
                logger.info("🛑 Interrupted by user")
                break
            
            except Exception as e:
                logger.error(f"❌ Connection error: {e}")
                reconnect_attempts += 1
                
                # Persist buffer before reconnecting
                try:
                    self._persist()
                except Exception:
                    pass
                
                # Exponential backoff
                wait_time = min(60, 2 ** reconnect_attempts)
                logger.info(f"⏳ Reconnecting in {wait_time}s ({reconnect_attempts}/{max_reconnects})...")
                time.sleep(wait_time)
            
            finally:
                if self.ws:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
        
        # Final persist
        try:
            self._persist()
        except Exception:
            pass
        
        logger.error(f"❌ Max reconnection attempts ({max_reconnects}) reached")

    def _listen(self):
        """Listen for messages on the WebSocket."""
        while self.ws:
            try:
                message = self.ws.recv()
                if not message:
                    break
                
                try:
                    payload = json.loads(message)
                except Exception:
                    continue
                
                # Handle status messages
                if isinstance(payload, dict):
                    if payload.get('ev') == 'status':
                        logger.info(f"📊 {payload.get('message')}")
                        continue
                
                # Handle tick arrays
                arr = payload if isinstance(payload, list) else [payload]
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    
                    record = {
                        't': item.get('t'),
                        'sip_timestamp': item.get('sip_timestamp'),
                        'p': item.get('p'),  # trade price
                        's': item.get('s'),  # trade size
                        'bid': item.get('b'),  # quote bid
                        'ask': item.get('a'),  # quote ask
                        'ev': item.get('ev'),  # event type
                    }
                    
                    # Only keep if has timestamp
                    if record.get('t') or record.get('sip_timestamp'):
                        self.buffer.append(record)
                        self.tick_count += 1
                        
                        if self.tick_count % 100 == 0:
                            logger.info(f"📍 Received {self.tick_count} total ticks")
                
                # Flush buffer periodically
                if len(self.buffer) >= 100 or (time.time() - self.last_flush) > 30:
                    self._persist()
                    
            except Exception as e:
                logger.error(f"❌ Listen error: {e}")
                break



def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='C:XAUUSD', help='Symbol to collect (default: C:XAUUSD)')
    parser.add_argument('--api-key', default=None, help='Polygon API key (uses POLYGON_API_KEY env if not provided)')
    parser.add_argument('--url', default=None, help='WebSocket URL (default: wss://socket.polygon.io/stocks)')
    
    args = parser.parse_args()
    
    logger.info(f"🎯 Starting Polygon WebSocket collector")
    logger.info(f"   Symbol: {args.symbol}")
    logger.info(f"   Output: {DEFAULT_OUTPUT}")
    logger.info(f"   Ctrl+C to stop\n")
    
    collector = PolygonWSCollector(api_key=args.api_key, symbol=args.symbol)
    collector.run(url=args.url)


if __name__ == '__main__':
    main()
