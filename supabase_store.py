"""
Supabase Integration
====================

Signal persistence and retrieval from Supabase
"""

import logging
from typing import List, Dict, Optional
import json
from datetime import datetime, timezone

try:
    from supabase import create_client, Client
except ImportError:
    Client = None
    logging.warning("supabase-py not installed")

from live_signal_engine import Config, Signal

logger = logging.getLogger(__name__)


class SupabaseSignalStore:
    """Stores and retrieves signals from Supabase."""
    
    def __init__(self):
        """Initialize Supabase client."""
        if not Client:
            raise ImportError("supabase-py required: pip install supabase")
        
        self.url = Config.SUPABASE_URL
        self.key = Config.SUPABASE_KEY
        self.table = Config.SUPABASE_TABLE
        
        self.client: Optional[Client] = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Supabase."""
        try:
            self.client = create_client(self.url, self.key)
            logger.info("✓ Connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self.client = None
    
    def store_signal(self, signal: Signal) -> bool:
        """
        Store a signal in Supabase.
        
        Args:
            signal: Signal to store
            
        Returns:
            True if successful
        """
        if not self.client:
            logger.warning("Supabase not connected")
            return False
        
        try:
            record = {
                'symbol': signal.symbol,
                'timeframe': signal.timeframe,
                'timestamp': signal.timestamp.isoformat(),
                'signal': signal.signal,
                'confidence': signal.confidence,
                'model_output': json.dumps(signal.model_output),
                'features': json.dumps(signal.features) if signal.features else None,
                'created_at': datetime.now(timezone.utc).isoformat(),
            }
            
            response = self.client.table(self.table).insert(record).execute()
            
            if response.data:
                logger.info(f"✓ Stored signal to Supabase: {signal.symbol} {signal.timeframe}")
                return True
            else:
                logger.error(f"Supabase insert returned no data: {response}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
            return False
    
    def store_signals_batch(self, signals: List[Signal]) -> int:
        """
        Store multiple signals.
        
        Args:
            signals: List of signals
            
        Returns:
            Number of successfully stored signals
        """
        count = 0
        for signal in signals:
            if self.store_signal(signal):
                count += 1
        return count
    
    def get_latest_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """
        Get latest signals.
        
        Args:
            symbol: Optional symbol filter
            limit: Max results
            
        Returns:
            List of signal records
        """
        if not self.client:
            logger.warning("Supabase not connected")
            return []
        
        try:
            query = self.client.table(self.table).select("*")
            
            if symbol:
                query = query.eq("symbol", symbol)
            
            response = query.order("created_at", desc=True).limit(limit).execute()
            
            return response.data if response.data else []
        
        except Exception as e:
            logger.error(f"Failed to fetch signals: {e}")
            return []
    
    def get_signals_by_timeframe(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get latest signals for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe ("1T", "5T", etc.)
            limit: Max results
            
        Returns:
            List of signal records
        """
        if not self.client:
            return []
        
        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq("symbol", symbol)
                .eq("timeframe", timeframe)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            
            return response.data if response.data else []
        
        except Exception as e:
            logger.error(f"Failed to fetch signals: {e}")
            return []
    
    def create_table_if_not_exists(self) -> bool:
        """
        Create signals table if it doesn't exist.
        
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        # Note: This would require direct SQL access or migrations
        # For now, assume table is created manually
        logger.info("Signals table structure (create manually if needed):")
        logger.info("""
            CREATE TABLE signals (
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                signal INTEGER NOT NULL CHECK (signal IN (-1, 0, 1)),
                confidence FLOAT NOT NULL,
                model_output JSONB,
                features JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(symbol, timeframe, timestamp)
            );
        """)
        
        return True


class SignalBroadcaster:
    """Broadcasts signals to real-time listeners (Supabase Realtime, etc.)"""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        """
        Args:
            supabase_client: Optional Supabase client for realtime
        """
        self.supabase_client = supabase_client
        self.listeners = []
    
    def subscribe(self, callback) -> None:
        """Subscribe to signal broadcasts."""
        self.listeners.append(callback)
    
    def broadcast(self, signal: Signal) -> None:
        """Broadcast signal to all listeners."""
        for listener in self.listeners:
            try:
                listener(signal)
            except Exception as e:
                logger.error(f"Listener error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    store = SupabaseSignalStore()
    
    # Create test signal
    from live_signal_engine import Signal
    import pandas as pd
    
    test_signal = Signal(
        symbol="XAUUSD",
        timeframe="5T",
        timestamp=pd.Timestamp.now(tz='UTC'),
        signal=1,
        confidence=0.85,
    )
    
    # Try to store
    store.store_signal(test_signal)
    
    # Retrieve
    signals = store.get_latest_signals("XAUUSD", limit=10)
    for sig in signals:
        print(sig)
