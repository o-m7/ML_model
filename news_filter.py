#!/usr/bin/env python3
"""
NEWS-BASED EVENT FILTER
========================
Avoid trading around major economic events (NFP, FOMC, CPI, etc.)
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None


# Currency mapping for symbols
SYMBOL_CURRENCIES = {
    'EURUSD': ['EUR', 'USD'],
    'GBPUSD': ['GBP', 'USD'],
    'AUDUSD': ['AUD', 'USD'],
    'NZDUSD': ['NZD', 'USD'],
    'XAUUSD': ['USD'],  # Gold priced in USD
    'XAGUSD': ['USD'],  # Silver priced in USD
}

# Default blackout windows (minutes before/after event)
BLACKOUT_WINDOWS = {
    'high': {'before': 30, 'after': 30},      # High impact: 30 min before/after
    'medium': {'before': 15, 'after': 15},    # Medium impact: 15 min before/after
    'low': {'before': 5, 'after': 5},         # Low impact: 5 min before/after
}


class NewsFilter:
    """
    Filter for avoiding trades around major economic events.
    """
    
    def __init__(self, use_supabase: bool = True):
        """
        Initialize news filter.
        
        Args:
            use_supabase: If True, load events from Supabase. If False, use hardcoded events.
        """
        self.use_supabase = use_supabase and supabase is not None
        self.events_cache = []
        self.cache_expiry = None
    
    def load_events_from_supabase(self, hours_ahead: int = 24) -> List[Dict]:
        """
        Load upcoming events from Supabase.
        
        Args:
            hours_ahead: How many hours ahead to load events
        
        Returns:
            List of event dictionaries
        """
        if not self.use_supabase:
            return []
        
        try:
            now = datetime.now(timezone.utc)
            end_time = now + timedelta(hours=hours_ahead)
            
            response = supabase.table('economic_events').select('*').gte(
                'event_time', now.isoformat()
            ).lte(
                'event_time', end_time.isoformat()
            ).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            print(f"⚠️  Failed to load events from Supabase: {e}")
            return []
    
    def get_hardcoded_events(self) -> List[Dict]:
        """
        Get hardcoded major economic events.
        Useful as fallback when Supabase is unavailable.
        """
        # This would be populated with known recurring events
        # For now, return empty list (will be populated by fetch_economic_calendar.py)
        return []
    
    def get_events(self, hours_ahead: int = 24, force_refresh: bool = False) -> List[Dict]:
        """
        Get economic events, using cache if available.
        
        Args:
            hours_ahead: How many hours ahead to look
            force_refresh: Force refresh from Supabase
        
        Returns:
            List of event dictionaries
        """
        now = datetime.now(timezone.utc)
        
        # Check cache validity
        if not force_refresh and self.cache_expiry and now < self.cache_expiry:
            return self.events_cache
        
        # Refresh cache
        if self.use_supabase:
            events = self.load_events_from_supabase(hours_ahead)
        else:
            events = self.get_hardcoded_events()
        
        self.events_cache = events
        self.cache_expiry = now + timedelta(minutes=15)  # Cache for 15 minutes
        
        return events
    
    def is_in_blackout_window(self, symbol: str, check_time: Optional[datetime] = None) -> bool:
        """
        Check if current time is within blackout window for any relevant event.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            check_time: Time to check (default: now)
        
        Returns:
            True if in blackout window, False otherwise
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        
        # Ensure check_time is timezone-aware
        if check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)
        
        # Get relevant currencies for this symbol
        currencies = SYMBOL_CURRENCIES.get(symbol, [])
        if not currencies:
            return False  # Unknown symbol, allow trading
        
        # Get upcoming events
        events = self.get_events(hours_ahead=2)  # Check 2 hours ahead
        
        for event in events:
            # Check if event affects this symbol
            event_currency = event.get('currency', '')
            if event_currency not in currencies:
                continue
            
            # Parse event time
            event_time_str = event.get('event_time', '')
            if not event_time_str:
                continue
            
            try:
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
            except:
                continue
            
            # Get blackout window for this event impact level
            impact = event.get('impact', 'medium').lower()
            window = BLACKOUT_WINDOWS.get(impact, BLACKOUT_WINDOWS['medium'])
            
            # Calculate blackout start/end
            blackout_start = event_time - timedelta(minutes=window['before'])
            blackout_end = event_time + timedelta(minutes=window['after'])
            
            # Check if check_time is in blackout window
            if blackout_start <= check_time <= blackout_end:
                return True
        
        return False
    
    def get_next_event(self, symbol: str, hours_ahead: int = 24) -> Optional[Dict]:
        """
        Get the next relevant economic event for a symbol.
        
        Args:
            symbol: Trading symbol
            hours_ahead: How many hours ahead to look
        
        Returns:
            Next event dictionary or None
        """
        currencies = SYMBOL_CURRENCIES.get(symbol, [])
        if not currencies:
            return None
        
        events = self.get_events(hours_ahead=hours_ahead)
        
        # Filter events for relevant currencies
        relevant_events = [
            e for e in events
            if e.get('currency', '') in currencies
        ]
        
        # Sort by event time
        relevant_events.sort(key=lambda e: e.get('event_time', ''))
        
        return relevant_events[0] if relevant_events else None
    
    def get_blackout_periods(self, symbol: str, hours_ahead: int = 24) -> List[Dict]:
        """
        Get all blackout periods for a symbol.
        
        Args:
            symbol: Trading symbol
            hours_ahead: How many hours ahead to look
        
        Returns:
            List of blackout period dictionaries with 'start', 'end', 'event'
        """
        currencies = SYMBOL_CURRENCIES.get(symbol, [])
        if not currencies:
            return []
        
        events = self.get_events(hours_ahead=hours_ahead)
        blackout_periods = []
        
        for event in events:
            if event.get('currency', '') not in currencies:
                continue
            
            event_time_str = event.get('event_time', '')
            if not event_time_str:
                continue
            
            try:
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
            except:
                continue
            
            impact = event.get('impact', 'medium').lower()
            window = BLACKOUT_WINDOWS.get(impact, BLACKOUT_WINDOWS['medium'])
            
            blackout_start = event_time - timedelta(minutes=window['before'])
            blackout_end = event_time + timedelta(minutes=window['after'])
            
            blackout_periods.append({
                'start': blackout_start,
                'end': blackout_end,
                'event': event.get('event_name', 'Unknown Event'),
                'impact': impact,
                'currency': event.get('currency', '')
            })
        
        return blackout_periods


def test_news_filter():
    """Test news filter functionality."""
    print("\n" + "="*80)
    print("TESTING NEWS FILTER")
    print("="*80 + "\n")
    
    # Initialize filter
    news_filter = NewsFilter(use_supabase=True)
    
    # Test symbols
    test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
    
    for symbol in test_symbols:
        print(f"\n--- {symbol} ---")
        
        # Check if in blackout
        in_blackout = news_filter.is_in_blackout_window(symbol)
        print(f"In blackout: {'YES ❌' if in_blackout else 'NO ✅'}")
        
        # Get next event
        next_event = news_filter.get_next_event(symbol, hours_ahead=48)
        if next_event:
            print(f"Next event: {next_event.get('event_name', 'Unknown')}")
            print(f"  Time: {next_event.get('event_time', 'Unknown')}")
            print(f"  Impact: {next_event.get('impact', 'Unknown')}")
        else:
            print("Next event: None in next 48 hours")
        
        # Get blackout periods
        blackouts = news_filter.get_blackout_periods(symbol, hours_ahead=48)
        if blackouts:
            print(f"Blackout periods: {len(blackouts)}")
            for i, blackout in enumerate(blackouts[:3], 1):  # Show first 3
                print(f"  {i}. {blackout['event']} ({blackout['impact']})")
                print(f"     {blackout['start'].strftime('%Y-%m-%d %H:%M')} to {blackout['end'].strftime('%H:%M')}")
        else:
            print("Blackout periods: None in next 48 hours")
    
    print("\n" + "="*80)
    print("✅ Test complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_news_filter()

