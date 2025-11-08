#!/usr/bin/env python3
"""
ECONOMIC CALENDAR FETCHER
==========================
Fetches upcoming high-impact economic events and stores them in Supabase.
Uses free economic calendar data sources.
"""

import os
import sys
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from dotenv import load_dotenv
from supabase import create_client
import json

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Major economic events to track (manually curated for now)
# In production, you would scrape from ForexFactory, Investing.com, or use a paid API
RECURRING_EVENTS = {
    'USD': [
        {'name': 'NFP (Non-Farm Payrolls)', 'day_of_month': 'first_friday', 'time': '13:30'},
        {'name': 'FOMC Meeting', 'frequency': 'every_6_weeks', 'time': '19:00'},
        {'name': 'CPI (Inflation)', 'day_of_month': 12, 'time': '13:30'},
        {'name': 'Retail Sales', 'day_of_month': 15, 'time': '13:30'},
        {'name': 'GDP', 'frequency': 'quarterly', 'time': '13:30'},
        {'name': 'Fed Chair Speech', 'frequency': 'irregular', 'time': 'varies'},
    ],
    'EUR': [
        {'name': 'ECB Interest Rate Decision', 'frequency': 'every_6_weeks', 'time': '12:45'},
        {'name': 'ECB Press Conference', 'frequency': 'every_6_weeks', 'time': '13:30'},
        {'name': 'Eurozone CPI', 'day_of_month': 'end_of_month', 'time': '10:00'},
        {'name': 'Eurozone GDP', 'frequency': 'quarterly', 'time': '10:00'},
    ],
    'GBP': [
        {'name': 'BOE Interest Rate Decision', 'frequency': 'every_6_weeks', 'time': '12:00'},
        {'name': 'UK CPI', 'day_of_month': 15, 'time': '07:00'},
        {'name': 'UK GDP', 'frequency': 'quarterly', 'time': '07:00'},
    ],
    'AUD': [
        {'name': 'RBA Interest Rate Decision', 'day_of_month': 'first_tuesday', 'time': '03:30'},
        {'name': 'Australia CPI', 'frequency': 'quarterly', 'time': '00:30'},
    ],
    'NZD': [
        {'name': 'RBNZ Interest Rate Decision', 'frequency': 'quarterly', 'time': '01:00'},
    ],
}


def generate_dummy_events(days_ahead: int = 7) -> List[Dict]:
    """
    Generate sample economic events for testing.
    In production, replace with actual calendar scraper.
    """
    events = []
    now = datetime.now(timezone.utc)
    
    # For demo: Create sample events
    sample_events = [
        {
            'currency': 'USD',
            'event_name': 'NFP (Non-Farm Payrolls)',
            'impact': 'high',
            'days_from_now': 2,
            'time': '13:30'
        },
        {
            'currency': 'EUR',
            'event_name': 'ECB Interest Rate Decision',
            'impact': 'high',
            'days_from_now': 3,
            'time': '12:45'
        },
        {
            'currency': 'USD',
            'event_name': 'FOMC Meeting',
            'impact': 'high',
            'days_from_now': 5,
            'time': '19:00'
        },
        {
            'currency': 'GBP',
            'event_name': 'BOE Interest Rate Decision',
            'impact': 'high',
            'days_from_now': 4,
            'time': '12:00'
        },
    ]
    
    for event in sample_events:
        if event['days_from_now'] <= days_ahead:
            event_date = now + timedelta(days=event['days_from_now'])
            event_time = event_date.replace(
                hour=int(event['time'].split(':')[0]),
                minute=int(event['time'].split(':')[1]),
                second=0,
                microsecond=0
            )
            
            # Calculate blackout windows
            blackout_start = event_time - timedelta(minutes=30)
            blackout_end = event_time + timedelta(minutes=30)
            
            events.append({
                'currency': event['currency'],
                'event_name': event['event_name'],
                'impact': event['impact'],
                'event_time': event_time.isoformat(),
                'blackout_start': blackout_start.isoformat(),
                'blackout_end': blackout_end.isoformat(),
            })
    
    return events


def scrape_forexfactory(days_ahead: int = 7) -> List[Dict]:
    """
    Scrape economic calendar from ForexFactory.
    NOTE: This is a simplified example. In production, use proper scraping or paid API.
    """
    # For now, return dummy events
    # TODO: Implement actual ForexFactory scraping or use a paid calendar API
    print("  ℹ️  Using sample events (ForexFactory scraping not yet implemented)")
    return generate_dummy_events(days_ahead)


def store_events_in_supabase(events: List[Dict]) -> int:
    """Store events in Supabase, avoiding duplicates."""
    stored_count = 0
    
    for event in events:
        try:
            # Check if event already exists
            existing = supabase.table('economic_events').select('*').eq(
                'currency', event['currency']
            ).eq(
                'event_name', event['event_name']
            ).eq(
                'event_time', event['event_time']
            ).execute()
            
            if existing.data:
                print(f"  ⏭️  Skipping duplicate: {event['currency']} {event['event_name']}")
                continue
            
            # Insert new event
            supabase.table('economic_events').insert(event).execute()
            print(f"  ✅ Stored: {event['currency']} {event['event_name']} at {event['event_time']}")
            stored_count += 1
            
        except Exception as e:
            print(f"  ❌ Error storing {event['event_name']}: {e}")
    
    return stored_count


def cleanup_old_events():
    """Delete events older than 1 day."""
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        
        result = supabase.table('economic_events').delete().lt(
            'event_time', cutoff
        ).execute()
        
        print(f"  🗑️  Cleaned up old events")
        
    except Exception as e:
        print(f"  ⚠️  Error cleaning up: {e}")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print(f"ECONOMIC CALENDAR UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Cleanup old events
    cleanup_old_events()
    
    # Fetch upcoming events
    print("\n📅 Fetching upcoming economic events...")
    events = scrape_forexfactory(days_ahead=7)
    
    print(f"\n  Found {len(events)} events")
    
    # Store in Supabase
    print("\n💾 Storing in Supabase...")
    stored = store_events_in_supabase(events)
    
    print("\n" + "="*80)
    print(f"✅ Stored {stored} new events")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
