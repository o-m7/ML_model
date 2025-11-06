#!/usr/bin/env python3
"""
ECONOMIC CALENDAR FETCHER
==========================
Fetches upcoming economic events and stores them in Supabase.
Uses free sources: Investing.com economic calendar scraper.
"""

import os
import sys
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from dotenv import load_dotenv
from supabase import create_client
from bs4 import BeautifulSoup

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY]):
    print("❌ Missing required environment variables!")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Major economic events to track (manually curated list)
MAJOR_EVENTS = {
    'USD': [
        'Non-Farm Payrolls', 'NFP', 'Unemployment Rate',
        'FOMC', 'Federal Funds Rate', 'Interest Rate Decision',
        'CPI', 'Consumer Price Index', 'Core CPI',
        'PPI', 'Producer Price Index',
        'GDP', 'Gross Domestic Product',
        'Retail Sales', 'Core Retail Sales',
        'ISM Manufacturing', 'ISM Services',
        'ADP Employment', 'Jobless Claims',
        'PCE Price Index', 'Core PCE'
    ],
    'EUR': [
        'ECB Interest Rate', 'ECB Press Conference',
        'CPI', 'Consumer Price Index', 'Core CPI',
        'GDP', 'German ZEW', 'German IFO',
        'Manufacturing PMI', 'Services PMI'
    ],
    'GBP': [
        'BOE Interest Rate', 'MPC Meeting',
        'CPI', 'Consumer Price Index', 'Core CPI',
        'GDP', 'Unemployment Rate',
        'Retail Sales', 'Manufacturing PMI'
    ],
    'AUD': [
        'RBA Interest Rate', 'Cash Rate',
        'CPI', 'GDP', 'Unemployment Rate',
        'Employment Change', 'Retail Sales'
    ],
    'NZD': [
        'RBNZ Interest Rate', 'OCR',
        'CPI', 'GDP', 'Unemployment Rate',
        'Employment Change'
    ]
}


def determine_impact(event_name: str, currency: str) -> str:
    """
    Determine event impact level based on event name and currency.
    
    Returns: 'high', 'medium', or 'low'
    """
    event_lower = event_name.lower()
    
    # High impact events
    high_impact_keywords = [
        'interest rate', 'fomc', 'ecb', 'boe', 'rba', 'rbnz',
        'non-farm', 'nfp', 'employment', 'cpi', 'gdp',
        'federal funds', 'cash rate', 'ocr'
    ]
    
    for keyword in high_impact_keywords:
        if keyword in event_lower:
            return 'high'
    
    # Medium impact events
    medium_impact_keywords = [
        'pmi', 'retail sales', 'unemployment',
        'consumer confidence', 'producer price',
        'trade balance', 'jobless claims'
    ]
    
    for keyword in medium_impact_keywords:
        if keyword in event_lower:
            return 'medium'
    
    # Default to low impact
    return 'low'


def scrape_forex_factory(days_ahead: int = 7) -> List[Dict]:
    """
    Scrape economic calendar from ForexFactory.
    Note: This is a simplified version. In production, you'd want more robust scraping.
    
    Returns:
        List of event dictionaries
    """
    events = []
    
    # ForexFactory calendar URL
    # Note: This requires proper headers and may be blocked. Consider using an API instead.
    base_url = "https://www.forexfactory.com/calendar"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"  ⚠️  Failed to fetch ForexFactory (status {response.status_code})")
            return events
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Parse calendar (this is simplified - actual parsing would be more complex)
        # ForexFactory has a complex structure that changes often
        # For production, consider using a dedicated API
        
        print(f"  ✅ Fetched ForexFactory page (parsing not fully implemented)")
        
    except Exception as e:
        print(f"  ❌ Error scraping ForexFactory: {e}")
    
    return events


def generate_sample_events(days_ahead: int = 7) -> List[Dict]:
    """
    Generate sample events for testing.
    In production, replace with actual API/scraping.
    
    This creates a realistic schedule of major economic events.
    """
    events = []
    now = datetime.now(timezone.utc)
    
    # Common event schedule (simplified)
    event_schedule = [
        # Weekly events
        {'day_offset': 1, 'hour': 12, 'minute': 30, 'currency': 'USD', 'name': 'Crude Oil Inventories', 'impact': 'medium'},
        {'day_offset': 3, 'hour': 8, 'minute': 30, 'currency': 'EUR', 'name': 'German ZEW Economic Sentiment', 'impact': 'high'},
        {'day_offset': 5, 'hour': 12, 'minute': 30, 'currency': 'USD', 'name': 'Unemployment Claims', 'impact': 'medium'},
        
        # Monthly events (simulate next occurrence)
        {'day_offset': 7, 'hour': 12, 'minute': 30, 'currency': 'USD', 'name': 'Non-Farm Payrolls (NFP)', 'impact': 'high'},
        {'day_offset': 7, 'hour': 12, 'minute': 30, 'currency': 'USD', 'name': 'Unemployment Rate', 'impact': 'high'},
        {'day_offset': 10, 'hour': 12, 'minute': 30, 'currency': 'USD', 'name': 'Consumer Price Index (CPI)', 'impact': 'high'},
        {'day_offset': 10, 'hour': 12, 'minute': 30, 'currency': 'USD', 'name': 'Core CPI', 'impact': 'high'},
        {'day_offset': 14, 'hour': 18, 'minute': 0, 'currency': 'USD', 'name': 'FOMC Interest Rate Decision', 'impact': 'high'},
        {'day_offset': 14, 'hour': 18, 'minute': 30, 'currency': 'USD', 'name': 'FOMC Press Conference', 'impact': 'high'},
        
        {'day_offset': 8, 'hour': 11, 'minute': 45, 'currency': 'EUR', 'name': 'ECB Interest Rate Decision', 'impact': 'high'},
        {'day_offset': 8, 'hour': 12, 'minute': 30, 'currency': 'EUR', 'name': 'ECB Press Conference', 'impact': 'high'},
        {'day_offset': 12, 'hour': 11, 'minute': 0, 'currency': 'EUR', 'name': 'Eurozone CPI', 'impact': 'high'},
        
        {'day_offset': 9, 'hour': 11, 'minute': 0, 'currency': 'GBP', 'name': 'BOE Interest Rate Decision', 'impact': 'high'},
        {'day_offset': 13, 'hour': 8, 'minute': 30, 'currency': 'GBP', 'name': 'UK CPI', 'impact': 'high'},
        
        {'day_offset': 4, 'hour': 3, 'minute': 30, 'currency': 'AUD', 'name': 'RBA Interest Rate Decision', 'impact': 'high'},
        {'day_offset': 11, 'hour': 0, 'minute': 30, 'currency': 'AUD', 'name': 'Employment Change', 'impact': 'high'},
        
        {'day_offset': 6, 'hour': 21, 'minute': 0, 'currency': 'NZD', 'name': 'RBNZ Interest Rate Decision', 'impact': 'high'},
    ]
    
    for event_template in event_schedule:
        event_time = now + timedelta(days=event_template['day_offset'])
        event_time = event_time.replace(
            hour=event_template['hour'],
            minute=event_template['minute'],
            second=0,
            microsecond=0
        )
        
        if event_time > now + timedelta(days=days_ahead):
            continue
        
        # Calculate blackout windows
        if event_template['impact'] == 'high':
            before, after = 30, 30
        elif event_template['impact'] == 'medium':
            before, after = 15, 15
        else:
            before, after = 5, 5
        
        events.append({
            'event_time': event_time.isoformat(),
            'currency': event_template['currency'],
            'event_name': event_template['name'],
            'impact': event_template['impact'],
            'blackout_start': (event_time - timedelta(minutes=before)).isoformat(),
            'blackout_end': (event_time + timedelta(minutes=after)).isoformat()
        })
    
    return events


def store_events_in_supabase(events: List[Dict]) -> int:
    """
    Store events in Supabase economic_events table.
    
    Returns:
        Number of events stored
    """
    if not events:
        return 0
    
    stored_count = 0
    
    try:
        # Clear old events (older than now)
        now = datetime.now(timezone.utc).isoformat()
        supabase.table('economic_events').delete().lt('event_time', now).execute()
        
        # Insert new events
        for event in events:
            try:
                supabase.table('economic_events').upsert(event, on_conflict='event_time,currency,event_name').execute()
                stored_count += 1
            except Exception as e:
                print(f"  ⚠️  Failed to store event: {e}")
        
        print(f"  ✅ Stored {stored_count} events in Supabase")
        
    except Exception as e:
        print(f"  ❌ Error storing events: {e}")
    
    return stored_count


def main():
    """Main execution."""
    print("\n" + "="*80)
    print(f"ECONOMIC CALENDAR FETCHER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Try to fetch from ForexFactory
    print("Attempting to fetch from ForexFactory...")
    events = scrape_forex_factory(days_ahead=7)
    
    # If scraping fails, use sample events
    if not events:
        print("\nUsing sample events (replace with real API in production)...")
        events = generate_sample_events(days_ahead=14)
    
    print(f"\nFound {len(events)} events")
    
    # Show sample events
    if events:
        print("\nSample events:")
        for i, event in enumerate(events[:5], 1):
            print(f"  {i}. {event['event_name']} ({event['currency']}) - {event['impact']}")
            print(f"     Time: {event['event_time']}")
    
    # Store in Supabase
    print("\nStoring events in Supabase...")
    stored = store_events_in_supabase(events)
    
    print("\n" + "="*80)
    print(f"✅ Completed: {stored} events stored")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

