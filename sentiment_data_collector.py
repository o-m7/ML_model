#!/usr/bin/env python3
"""
SENTIMENT DATA COLLECTOR
=========================
Collects sentiment data hourly and stores in Supabase.
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

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Symbols to track
SYMBOLS = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']


def store_sentiment(sentiment_data: dict):
    """Store sentiment data in Supabase."""
    try:
        supabase.table('sentiment_data').insert({
            'symbol': sentiment_data['symbol'],
            'timestamp': sentiment_data['timestamp'],
            'news_sentiment': sentiment_data['news_sentiment'],
            'reddit_sentiment': sentiment_data['reddit_sentiment'],
            'aggregate_sentiment': sentiment_data['aggregate_sentiment'],
        }).execute()
        
        print(f"  ✅ Stored sentiment for {sentiment_data['symbol']}")
    
    except Exception as e:
        print(f"  ❌ Error storing {sentiment_data['symbol']}: {e}")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print(f"SENTIMENT DATA COLLECTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    analyzer = SentimentAnalyzer()
    
    collected = 0
    for symbol in SYMBOLS:
        try:
            print(f"\n📊 Collecting sentiment for {symbol}...")
            
            result = analyzer.get_aggregate_sentiment(symbol)
            
            if result['aggregate_sentiment'] != 0.0:
                store_sentiment(result)
                collected += 1
            else:
                print(f"  ⏭️  Skipping {symbol} - no data")
        
        except Exception as e:
            print(f"  ❌ Error with {symbol}: {e}")
    
    print("\n" + "="*80)
    print(f"✅ Collected sentiment for {collected}/{len(SYMBOLS)} symbols")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
