#!/usr/bin/env python3
"""
NEWS-BASED EVENT FILTER
========================
Avoids trading around major economic events to reduce whipsaw risk.
"""

import os
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Symbol to currency mapping
SYMBOL_TO_CURRENCY = {
    'XAUUSD': ['USD'],  # Gold affected by USD
    'XAGUSD': ['USD'],  # Silver affected by USD
    'EURUSD': ['EUR', 'USD'],
    'GBPUSD': ['GBP', 'USD'],
    'AUDUSD': ['AUD', 'USD'],
    'NZDUSD': ['NZD', 'USD'],
}

# Blackout windows (minutes before and after event)
BLACKOUT_WINDOW_BEFORE = 30  # 30 minutes before
BLACKOUT_WINDOW_AFTER = 30   # 30 minutes after


def get_upcoming_events(currency: str, hours_ahead: int = 24) -> List[Dict]:
    """Get upcoming economic events for a currency from Supabase."""
    try:
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=hours_ahead)
        
        response = supabase.table('economic_events').select('*').eq(
            'currency', currency
        ).gte(
            'event_time', now.isoformat()
        ).lte(
            'event_time', future.isoformat()
        ).eq(
            'impact', 'high'  # Only high-impact events
        ).execute()
        
        return response.data if response.data else []
    
    except Exception as e:
        print(f"  ⚠️  Error fetching events for {currency}: {e}")
        return []


def is_in_blackout_window(symbol: str, check_time: Optional[datetime] = None) -> Dict:
    """
    Check if current time is within blackout window for any high-impact event.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        check_time: Time to check (default: now)
    
    Returns:
        Dict with keys:
            - is_blackout: bool
            - event: dict or None (the event causing blackout)
            - reason: str (explanation)
    """
    if check_time is None:
        check_time = datetime.now(timezone.utc)
    
    # Get relevant currencies for this symbol
    currencies = SYMBOL_TO_CURRENCY.get(symbol, [])
    if not currencies:
        return {'is_blackout': False, 'event': None, 'reason': 'Unknown symbol'}
    
    # Check events for all relevant currencies
    for currency in currencies:
        events = get_upcoming_events(currency, hours_ahead=2)
        
        for event in events:
            event_time = pd.to_datetime(event['event_time'], utc=True)
            blackout_start = event_time - timedelta(minutes=BLACKOUT_WINDOW_BEFORE)
            blackout_end = event_time + timedelta(minutes=BLACKOUT_WINDOW_AFTER)
            
            if blackout_start <= check_time <= blackout_end:
                time_to_event = (event_time - check_time).total_seconds() / 60
                
                if time_to_event > 0:
                    reason = f"{currency} {event['event_name']} in {abs(time_to_event):.0f}min"
                else:
                    reason = f"{currency} {event['event_name']} {abs(time_to_event):.0f}min ago"
                
                return {
                    'is_blackout': True,
                    'event': event,
                    'reason': reason,
                    'currency': currency,
                    'event_time': event_time
                }
    
    return {'is_blackout': False, 'event': None, 'reason': 'No high-impact events'}


def get_blackout_status_for_symbols(symbols: List[str]) -> Dict[str, Dict]:
    """Get blackout status for multiple symbols."""
    check_time = datetime.now(timezone.utc)
    results = {}
    
    for symbol in symbols:
        results[symbol] = is_in_blackout_window(symbol, check_time)
    
    return results


def test_news_filter():
    """Test the news filter."""
    print("\n" + "="*80)
    print("TESTING NEWS FILTER")
    print("="*80 + "\n")
    
    symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
    
    for symbol in symbols:
        result = is_in_blackout_window(symbol)
        
        if result['is_blackout']:
            print(f"🚫 {symbol}: BLACKOUT - {result['reason']}")
        else:
            print(f"✅ {symbol}: CLEAR - {result['reason']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_news_filter()
