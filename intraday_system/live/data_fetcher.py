"""Live data fetcher from Polygon API."""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PolygonDataFetcher:
    """Fetch live OHLCV data from Polygon REST API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon data fetcher.
        
        Args:
            api_key: Polygon API key (defaults to POLYGON_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
        
        self.base_url = "https://api.polygon.io/v2"
    
    def fetch_latest_bars(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int = 200,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Fetch latest bars for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD', 'EUR/USD')
            timeframe: Timeframe (5T, 15T, 30T, 1H, 2H, 4H)
            n_bars: Number of bars to fetch
            adjusted: Whether to use adjusted prices
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert symbol format for Polygon
        polygon_symbol = self._format_symbol(symbol)
        
        # Convert timeframe to Polygon format
        multiplier, timespan = self._parse_timeframe(timeframe)
        
        # Calculate date range
        end_time = datetime.now(timezone.utc)
        
        # Estimate start time (add buffer for non-trading hours)
        bars_per_day = self._estimate_bars_per_day(timeframe)
        days_needed = int(n_bars / bars_per_day * 1.5) + 1  # 1.5x buffer
        start_time = end_time - timedelta(days=days_needed)
        
        # Format dates for API
        start_str = start_time.strftime('%Y-%m-%d')
        end_str = end_time.strftime('%Y-%m-%d')
        
        # Build URL
        url = f"{self.base_url}/aggs/ticker/{polygon_symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
        
        params = {
            'adjusted': str(adjusted).lower(),
            'sort': 'asc',
            'limit': 50000,
            'apiKey': self.api_key
        }
        
        # Make request
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'OK':
                raise ValueError(f"Polygon API error: {data.get('error', 'Unknown error')}")
            
            results = data.get('results', [])
            if not results:
                raise ValueError(f"No data returned for {symbol} {timeframe}")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Rename columns
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'n': 'transactions'
            })
            
            # Convert timestamp (milliseconds to datetime)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Select required columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Sort and take last n_bars
            df = df.sort_values('timestamp').tail(n_bars).reset_index(drop=True)
            
            print(f"✓ Fetched {len(df)} bars for {symbol} {timeframe}")
            print(f"  Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch data from Polygon: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing Polygon data: {e}")
    
    def _format_symbol(self, symbol: str) -> str:
        """
        Convert symbol to Polygon format.
        
        Examples:
            XAUUSD -> C:XAUUSD (forex/crypto)
            EURUSD -> C:EURUSD
            AAPL -> AAPL (stock)
        """
        # Forex/Metal pairs
        forex_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD',
            'AUDUSD', 'NZDUSD', 'USDCHF', 'XAUUSD', 'XAGUSD'
        ]
        
        if symbol in forex_pairs:
            return f"C:{symbol}"
        
        # Otherwise assume it's a stock ticker
        return symbol
    
    def _parse_timeframe(self, timeframe: str) -> tuple:
        """
        Parse timeframe string to Polygon format.
        
        Args:
            timeframe: '5T', '15T', '30T', '1H', '2H', '4H'
            
        Returns:
            (multiplier, timespan) e.g., (5, 'minute') or (1, 'hour')
        """
        timeframe = timeframe.upper()
        
        if timeframe.endswith('T'):
            # Minutes
            multiplier = int(timeframe[:-1])
            return (multiplier, 'minute')
        elif timeframe.endswith('H'):
            # Hours
            multiplier = int(timeframe[:-1])
            return (multiplier, 'hour')
        elif timeframe.endswith('D'):
            # Days
            multiplier = int(timeframe[:-1])
            return (multiplier, 'day')
        else:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")
    
    def _estimate_bars_per_day(self, timeframe: str) -> int:
        """Estimate number of bars per trading day."""
        tf_upper = timeframe.upper()
        
        if tf_upper == '5T':
            return 288  # 24 hours for forex
        elif tf_upper == '15T':
            return 96
        elif tf_upper == '30T':
            return 48
        elif tf_upper == '1H':
            return 24
        elif tf_upper == '2H':
            return 12
        elif tf_upper == '4H':
            return 6
        else:
            return 24  # Default


def get_live_data(
    symbol: str,
    timeframe: str,
    n_bars: int = 200,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to fetch live data.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        n_bars: Number of bars to fetch
        api_key: Optional Polygon API key
        
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = PolygonDataFetcher(api_key)
    return fetcher.fetch_latest_bars(symbol, timeframe, n_bars)


# Example usage
if __name__ == '__main__':
    # Test fetching
    try:
        df = get_live_data('XAUUSD', '15T', n_bars=100)
        print("\nSample data:")
        print(df.tail())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error: {e}")

