#!/usr/bin/env python3
"""
SENTIMENT ANALYSIS ENGINE
==========================
Aggregates market sentiment from multiple free sources:
- NewsAPI (100 requests/day free)
- Reddit API (praw)
- Twitter API v2 (free tier)
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dotenv import load_dotenv
import re

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  vaderSentiment not installed. Install with: pip install vaderSentiment")

# News API
try:
    from newsapi import NewsApiClient
    NEWS_API_AVAILABLE = True
except ImportError:
    NEWS_API_AVAILABLE = False
    print("⚠️  newsapi-python not installed. Install with: pip install newsapi-python")

# Reddit API
try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    print("⚠️  praw not installed. Install with: pip install praw")

# Twitter API
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    print("⚠️  tweepy not installed. Install with: pip install tweepy")

load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'MLTradingBot/1.0')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

# Symbol keywords for search
SYMBOL_KEYWORDS = {
    'EURUSD': ['EUR/USD', 'euro dollar', 'eurusd', 'EURUSD'],
    'GBPUSD': ['GBP/USD', 'pound dollar', 'gbpusd', 'cable'],
    'AUDUSD': ['AUD/USD', 'aussie dollar', 'audusd'],
    'NZDUSD': ['NZD/USD', 'kiwi dollar', 'nzdusd'],
    'XAUUSD': ['gold', 'XAU/USD', 'gold price', 'gold trading'],
    'XAGUSD': ['silver', 'XAG/USD', 'silver price'],
}


class SentimentAnalyzer:
    """
    Multi-source sentiment analyzer for trading symbols.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Initialize APIs
        self.news_api = None
        self.reddit = None
        self.twitter_client = None
        
        if NEWS_API_AVAILABLE and NEWS_API_KEY:
            try:
                self.news_api = NewsApiClient(api_key=NEWS_API_KEY)
            except Exception as e:
                print(f"⚠️  Failed to initialize NewsAPI: {e}")
        
        if REDDIT_AVAILABLE and REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            try:
                self.reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize Reddit: {e}")
        
        if TWITTER_AVAILABLE and TWITTER_BEARER_TOKEN:
            try:
                self.twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
            except Exception as e:
                print(f"⚠️  Failed to initialize Twitter: {e}")
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of text using VADER.
        
        Returns:
            Sentiment score from -1 (bearish) to +1 (bullish)
        """
        if not self.vader or not text:
            return 0.0
        
        # Clean text
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#', '', text)  # Remove hashtags but keep words
        
        # Get VADER sentiment
        scores = self.vader.polarity_scores(text)
        
        # Return compound score (-1 to +1)
        return scores['compound']
    
    def get_news_sentiment(self, symbol: str, hours_back: int = 24) -> Optional[Dict]:
        """
        Get sentiment from news articles.
        
        Returns:
            Dict with sentiment score and article count
        """
        if not self.news_api:
            return None
        
        keywords = SYMBOL_KEYWORDS.get(symbol, [symbol])
        query = ' OR '.join(keywords)
        
        from_date = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
        
        try:
            # Search for articles
            response = self.news_api.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=20
            )
            
            articles = response.get('articles', [])
            
            if not articles:
                return {'sentiment': 0.0, 'count': 0, 'source': 'news'}
            
            # Analyze sentiment of titles and descriptions
            sentiments = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title} {description}"
                
                sentiment = self.analyze_text(text)
                sentiments.append(sentiment)
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            
            return {
                'sentiment': avg_sentiment,
                'count': len(articles),
                'source': 'news'
            }
            
        except Exception as e:
            print(f"  ⚠️  News sentiment error: {e}")
            return None
    
    def get_reddit_sentiment(self, symbol: str, hours_back: int = 24) -> Optional[Dict]:
        """
        Get sentiment from Reddit (r/Forex, r/wallstreetbets, r/algotrading).
        
        Returns:
            Dict with sentiment score and post count
        """
        if not self.reddit:
            return None
        
        keywords = SYMBOL_KEYWORDS.get(symbol, [symbol])
        subreddits = ['Forex', 'wallstreetbets', 'algotrading', 'options', 'stocks']
        
        sentiments = []
        post_count = 0
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts containing symbol keywords
                for keyword in keywords[:2]:  # Limit to first 2 keywords
                    try:
                        for post in subreddit.search(keyword, time_filter='day', limit=5):
                            text = f"{post.title} {post.selftext}"
                            sentiment = self.analyze_text(text)
                            sentiments.append(sentiment)
                            post_count += 1
                    except Exception as e:
                        continue
            
            if not sentiments:
                return {'sentiment': 0.0, 'count': 0, 'source': 'reddit'}
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                'sentiment': avg_sentiment,
                'count': post_count,
                'source': 'reddit'
            }
            
        except Exception as e:
            print(f"  ⚠️  Reddit sentiment error: {e}")
            return None
    
    def get_twitter_sentiment(self, symbol: str, hours_back: int = 24) -> Optional[Dict]:
        """
        Get sentiment from Twitter.
        
        Returns:
            Dict with sentiment score and tweet count
        """
        if not self.twitter_client:
            return None
        
        keywords = SYMBOL_KEYWORDS.get(symbol, [symbol])
        query = ' OR '.join(keywords[:2])  # Limit to first 2 keywords
        
        sentiments = []
        tweet_count = 0
        
        try:
            # Search recent tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=20,
                tweet_fields=['created_at', 'text']
            )
            
            if not tweets.data:
                return {'sentiment': 0.0, 'count': 0, 'source': 'twitter'}
            
            for tweet in tweets.data:
                text = tweet.text
                sentiment = self.analyze_text(text)
                sentiments.append(sentiment)
                tweet_count += 1
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            
            return {
                'sentiment': avg_sentiment,
                'count': tweet_count,
                'source': 'twitter'
            }
            
        except Exception as e:
            print(f"  ⚠️  Twitter sentiment error: {e}")
            return None
    
    def get_aggregate_sentiment(self, symbol: str, hours_back: int = 24) -> Dict:
        """
        Get aggregated sentiment from all sources.
        
        Returns:
            Dict with individual and aggregate sentiment scores
        """
        print(f"\n  🔍 Analyzing sentiment for {symbol}...")
        
        # Get sentiment from each source
        news_sent = self.get_news_sentiment(symbol, hours_back)
        reddit_sent = self.get_reddit_sentiment(symbol, hours_back)
        twitter_sent = self.get_twitter_sentiment(symbol, hours_back)
        
        # Compile results
        sentiments = []
        weights = []
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'news_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'twitter_sentiment': 0.0,
            'aggregate_sentiment': 0.0,
            'total_sources': 0
        }
        
        if news_sent:
            result['news_sentiment'] = news_sent['sentiment']
            sentiments.append(news_sent['sentiment'])
            weights.append(0.5)  # News gets highest weight
            result['total_sources'] += 1
            print(f"    📰 News: {news_sent['sentiment']:+.3f} ({news_sent['count']} articles)")
        
        if reddit_sent:
            result['reddit_sentiment'] = reddit_sent['sentiment']
            sentiments.append(reddit_sent['sentiment'])
            weights.append(0.3)  # Reddit gets medium weight
            result['total_sources'] += 1
            print(f"    🔴 Reddit: {reddit_sent['sentiment']:+.3f} ({reddit_sent['count']} posts)")
        
        if twitter_sent:
            result['twitter_sentiment'] = twitter_sent['sentiment']
            sentiments.append(twitter_sent['sentiment'])
            weights.append(0.2)  # Twitter gets lowest weight (noise)
            result['total_sources'] += 1
            print(f"    🐦 Twitter: {twitter_sent['sentiment']:+.3f} ({twitter_sent['count']} tweets)")
        
        # Calculate weighted average
        if sentiments:
            total_weight = sum(weights)
            weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
            result['aggregate_sentiment'] = weighted_sum / total_weight
        
        # Interpret sentiment
        sentiment_label = self.interpret_sentiment(result['aggregate_sentiment'])
        print(f"    📊 Aggregate: {result['aggregate_sentiment']:+.3f} ({sentiment_label})")
        
        return result
    
    def interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score as label."""
        if score > 0.3:
            return "Strongly Bullish"
        elif score > 0.1:
            return "Bullish"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.3:
            return "Bearish"
        else:
            return "Strongly Bearish"


def test_sentiment_analyzer():
    """Test sentiment analyzer."""
    print("\n" + "="*80)
    print("TESTING SENTIMENT ANALYZER")
    print("="*80)
    
    analyzer = SentimentAnalyzer()
    
    # Test symbols
    test_symbols = ['EURUSD', 'XAUUSD']
    
    for symbol in test_symbols:
        result = analyzer.get_aggregate_sentiment(symbol, hours_back=24)
        
        print(f"\n  Result for {symbol}:")
        print(f"    Aggregate Sentiment: {result['aggregate_sentiment']:+.3f}")
        print(f"    Sources Used: {result['total_sources']}")
    
    print("\n" + "="*80)
    print("✅ Test complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_sentiment_analyzer()

