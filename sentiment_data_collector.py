#!/usr/bin/env python3
"""
SENTIMENT DATA COLLECTOR
=========================
Collects sentiment data and stores in Supabase for later use.
Run hourly via cron or GitHub Actions.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).parent))
from sentiment_analyzer import SentimentAnalyzer

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY]):
    print("❌ Missing required environment variables!")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Symbols to track
SYMBOLS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'XAUUSD', 'XAGUSD']


def collect_and_store_sentiment():
    """Collect sentiment for all symbols and store in Supabase."""
    print("\n" + "="*80)
    print(f"SENTIMENT DATA COLLECTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    analyzer = SentimentAnalyzer()
    
    stored_count = 0
    failed_count = 0
    
    for symbol in SYMBOLS:
        try:
            # Get sentiment
            result = analyzer.get_aggregate_sentiment(symbol, hours_back=24)
            
            # Store in Supabase
            supabase.table('sentiment_data').insert({
                'symbol': symbol,
                'timestamp': result['timestamp'],
                'news_sentiment': result['news_sentiment'],
                'reddit_sentiment': result['reddit_sentiment'],
                'twitter_sentiment': result['twitter_sentiment'],
                'aggregate_sentiment': result['aggregate_sentiment']
            }).execute()
            
            stored_count += 1
            print(f"  ✅ {symbol}: {result['aggregate_sentiment']:+.3f}")
            
        except Exception as e:
            failed_count += 1
            print(f"  ❌ {symbol}: {e}")
    
    print("\n" + "="*80)
    print(f"✅ Stored: {stored_count}/{len(SYMBOLS)}")
    print(f"❌ Failed: {failed_count}/{len(SYMBOLS)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    collect_and_store_sentiment()

