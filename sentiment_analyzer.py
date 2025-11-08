#!/usr/bin/env python3
"""
SENTIMENT ANALYZER
===================
Aggregates market sentiment from multiple free sources:
- NewsAPI (news articles)
- Reddit (r/wallstreetbets, r/Forex, r/algotrading)
- Twitter/X (financial keywords)

Outputs: -1 (bearish) to +1 (bullish)
"""

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Optional imports (gracefully handle if not available)
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    print("⚠️  newsapi-python not installed. Run: pip install newsapi-python")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    print("⚠️  praw not installed. Run: pip install praw")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  vaderSentiment not installed. Run: pip install vaderSentiment")

load_dotenv()

# API Keys (from .env)
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'MLTradingBot/1.0')

# Symbol keywords for search
SYMBOL_KEYWORDS = {
    'XAUUSD': ['gold', 'xau', 'precious metals'],
    'XAGUSD': ['silver', 'xag', 'precious metals'],
    'EURUSD': ['euro', 'eur', 'european central bank', 'ecb'],
    'GBPUSD': ['pound', 'gbp', 'sterling', 'bank of england'],
    'AUDUSD': ['aussie', 'aud', 'australian dollar', 'rba'],
    'NZDUSD': ['kiwi', 'nzd', 'new zealand dollar', 'rbnz'],
}


class SentimentAnalyzer:
    """Aggregate sentiment from multiple sources."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        self.newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if (NEWSAPI_AVAILABLE and NEWSAPI_KEY) else None
        self.reddit = None
        
        if PRAW_AVAILABLE and REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            try:
                self.reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize Reddit: {e}")
    
    def analyze_text(self, text: str) -> float:
        """Analyze sentiment of a single text using VADER."""
        if not self.vader or not text:
            return 0.0
        
        scores = self.vader.polarity_scores(text)
        return scores['compound']  # -1 to +1
    
    def get_news_sentiment(self, symbol: str, hours_back: int = 24) -> Optional[Dict]:
        """Get sentiment from news articles."""
        if not self.newsapi:
            return None
        
        keywords = SYMBOL_KEYWORDS.get(symbol, [symbol])
        query = ' OR '.join(keywords)
        
        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            
            articles = self.newsapi.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=20
            )
            
            if not articles or 'articles' not in articles:
                return {'sentiment': 0.0, 'count': 0}
            
            sentiments = []
            for article in articles['articles']:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title}. {description}"
                
                sentiment = self.analyze_text(text)
                if sentiment != 0:  # Only count non-neutral
                    sentiments.append(sentiment)
            
            if not sentiments:
                return {'sentiment': 0.0, 'count': 0}
            
            return {
                'sentiment': sum(sentiments) / len(sentiments),
                'count': len(sentiments)
            }
        
        except Exception as e:
            print(f"  ⚠️  News API error for {symbol}: {e}")
            return None
    
    def get_reddit_sentiment(self, symbol: str, limit: int = 50) -> Optional[Dict]:
        """Get sentiment from Reddit posts."""
        if not self.reddit:
            return None
        
        keywords = SYMBOL_KEYWORDS.get(symbol, [symbol])
        
        try:
            sentiments = []
            
            # Search in relevant subreddits
            subreddits = ['wallstreetbets', 'Forex', 'algotrading', 'stocks']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for keywords in hot posts
                    for keyword in keywords:
                        for submission in subreddit.search(keyword, limit=limit//len(keywords)):
                            # Analyze title and selftext
                            text = f"{submission.title}. {submission.selftext}"
                            sentiment = self.analyze_text(text)
                            
                            if sentiment != 0:
                                sentiments.append(sentiment)
                except Exception as e:
                    continue
            
            if not sentiments:
                return {'sentiment': 0.0, 'count': 0}
            
            return {
                'sentiment': sum(sentiments) / len(sentiments),
                'count': len(sentiments)
            }
        
        except Exception as e:
            print(f"  ⚠️  Reddit error for {symbol}: {e}")
            return None
    
    def get_aggregate_sentiment(self, symbol: str) -> Dict:
        """Get aggregate sentiment from all sources."""
        results = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'news_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'aggregate_sentiment': 0.0,
            'sources_available': []
        }
        
        sentiments = []
        weights = []
        
        # Get news sentiment
        news_result = self.get_news_sentiment(symbol, hours_back=24)
        if news_result and news_result['count'] > 0:
            results['news_sentiment'] = news_result['sentiment']
            sentiments.append(news_result['sentiment'])
            weights.append(news_result['count'] * 0.6)  # News weighted higher
            results['sources_available'].append('news')
            print(f"  📰 News sentiment: {news_result['sentiment']:.3f} ({news_result['count']} articles)")
        
        # Get Reddit sentiment
        reddit_result = self.get_reddit_sentiment(symbol, limit=30)
        if reddit_result and reddit_result['count'] > 0:
            results['reddit_sentiment'] = reddit_result['sentiment']
            sentiments.append(reddit_result['sentiment'])
            weights.append(reddit_result['count'] * 0.4)  # Reddit weighted lower
            results['sources_available'].append('reddit')
            print(f"  🤖 Reddit sentiment: {reddit_result['sentiment']:.3f} ({reddit_result['count']} posts)")
        
        # Calculate weighted aggregate
        if sentiments and weights:
            total_weight = sum(weights)
            weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / total_weight
            results['aggregate_sentiment'] = weighted_sentiment
            
            # Interpret sentiment
            if weighted_sentiment > 0.2:
                interpretation = "BULLISH 🟢"
            elif weighted_sentiment < -0.2:
                interpretation = "BEARISH 🔴"
            else:
                interpretation = "NEUTRAL ⚪"
            
            print(f"  📊 Aggregate: {weighted_sentiment:.3f} → {interpretation}")
        else:
            print(f"  ⚠️  No sentiment data available for {symbol}")
        
        return results


def test_sentiment():
    """Test sentiment analyzer."""
    print("\n" + "="*80)
    print("TESTING SENTIMENT ANALYZER")
    print("="*80 + "\n")
    
    analyzer = SentimentAnalyzer()
    
    if not analyzer.vader:
        print("❌ VADER not available. Install: pip install vaderSentiment")
        return
    
    # Test with a few symbols
    for symbol in ['XAUUSD', 'EURUSD']:
        print(f"\n{'='*80}")
        print(f"Analyzing sentiment for {symbol}")
        print(f"{'='*80}")
        
        result = analyzer.get_aggregate_sentiment(symbol)
        
        print(f"\nResult: {result}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_sentiment()
