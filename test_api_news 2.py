#!/usr/bin/env python
# coding: utf-8

"""
Test script using public APIs and RSS feeds to get financial news.
"""

import requests
import json
import time
import argparse
from datetime import datetime
import re
from bs4 import BeautifulSoup

def get_user_agent():
    """Return a common user agent string."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

def get_company_name(ticker):
    """Map a ticker to a company name."""
    mapping = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "GOOG": "Google",
        "AMZN": "Amazon",
        "META": "Meta",
        "TSLA": "Tesla",
        "NVDA": "NVIDIA",
        "NFLX": "Netflix",
        "PYPL": "PayPal"
    }
    return mapping.get(ticker, ticker)

def get_marketaux_news(ticker, api_key=None):
    """
    Get news from MarketAux API (free tier).
    
    No API key needed for this test. We'll use limited endpoints.
    """
    print(f"Getting news for {ticker} from MarketAux API...")
    
    # Default endpoint with limited data (free)
    url = "https://api.marketaux.com/v1/news/sample"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Got status code {response.status_code}")
            return []
        
        data = response.json()
        
        if "data" not in data:
            print("Error: Unexpected API response format")
            return []
        
        news_items = data["data"]
        results = []
        company_name = get_company_name(ticker)
        
        # Filter for news related to the ticker or company name
        for item in news_items:
            title = item.get("title", "")
            description = item.get("description", "")
            
            # Check if the news is related to the requested ticker/company
            if (ticker.lower() in title.lower() or company_name.lower() in title.lower() or
                ticker.lower() in description.lower() or company_name.lower() in description.lower()):
                
                results.append({
                    "title": title,
                    "description": description,
                    "published_at": item.get("published_at", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", "")
                })
                print(f"- {title}")
        
        if not results:
            print(f"No news found for {ticker} in the sample data")
        
        return results
    
    except Exception as e:
        print(f"Error getting news from MarketAux: {e}")
        return []

def get_gnews(ticker):
    """
    Get news from Google News RSS feed.
    """
    print(f"Getting news for {ticker} from Google News...")
    
    company = get_company_name(ticker)
    # Use company name for better results, but include ticker
    query = f"{company} {ticker} stock"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        headers = get_user_agent()
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Got status code {response.status_code}")
            return []
        
        # Parse the RSS feed
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.findAll("item")
        
        if not items:
            print("No news items found")
            return []
        
        results = []
        for item in items[:10]:  # Limit to 10 items
            title = item.title.text
            link = item.link.text
            pub_date = item.pubDate.text if item.pubDate else "Unknown"
            description = item.description.text if item.description else ""
            
            results.append({
                "title": title,
                "link": link,
                "published": pub_date,
                "description": description
            })
            print(f"- {title}")
        
        # Save results to file
        with open(f"{ticker}_gnews.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {ticker}_gnews.json")
        
        return results
    
    except Exception as e:
        print(f"Error getting Google News: {e}")
        return []

def get_reddit_data(ticker):
    """
    Get data from Reddit about a ticker.
    """
    print(f"Getting Reddit posts for {ticker}...")
    
    # Use the JSON API for the stocks subreddit
    url = "https://www.reddit.com/r/stocks/search.json"
    params = {
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week"
    }
    
    try:
        headers = get_user_agent()
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Got status code {response.status_code}")
            return []
        
        data = response.json()
        
        if "data" not in data or "children" not in data["data"]:
            print("Error: Unexpected API response format")
            return []
        
        posts = data["data"]["children"]
        results = []
        
        for post in posts:
            post_data = post["data"]
            
            # Skip posts that don't mention the ticker in the title
            if ticker.lower() not in post_data["title"].lower():
                continue
            
            results.append({
                "title": post_data["title"],
                "created_utc": post_data["created_utc"],
                "score": post_data["score"],
                "num_comments": post_data["num_comments"],
                "url": f"https://www.reddit.com{post_data['permalink']}"
            })
            print(f"- {post_data['title']} (Score: {post_data['score']})")
        
        if not results:
            print(f"No relevant Reddit posts found for {ticker}")
        else:
            # Save results to file
            with open(f"{ticker}_reddit.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {ticker}_reddit.json")
        
        return results
    
    except Exception as e:
        print(f"Error getting Reddit data: {e}")
        return []

def analyze_sentiment(texts):
    """
    Simple keyword-based sentiment analysis.
    """
    positive_words = [
        "bullish", "growth", "gain", "profit", "positive", "up", "rise", "rising",
        "beat", "exceed", "outperform", "strong", "strength", "opportunity", "buy"
    ]
    
    negative_words = [
        "bearish", "decline", "fall", "loss", "negative", "down", "drop", "concern",
        "risk", "miss", "underperform", "weak", "weakness", "warning", "sell"
    ]
    
    # Count positive and negative words
    positive_count = 0
    negative_count = 0
    
    for text in texts:
        text_lower = text.lower()
        positive_count += sum(1 for word in positive_words if word in text_lower)
        negative_count += sum(1 for word in negative_words if word in text_lower)
    
    # Determine overall sentiment
    if positive_count > negative_count:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # Calculate relevance score (mock)
    relevance = min(1.0, max(0.0, (positive_count + negative_count) / (len(texts) * 3)))
    
    return {
        "sentiment": sentiment,
        "relevance": round(relevance, 2),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "positive_words": positive_count,
        "negative_words": negative_count,
        "total_texts": len(texts)
    }

def main():
    parser = argparse.ArgumentParser(description='Get financial news using public APIs and RSS feeds.')
    parser.add_argument('ticker', type=str, help='Stock ticker to get news for', nargs='?', default='AAPL')
    parser.add_argument('--source', '-s', type=str, choices=['marketaux', 'gnews', 'reddit', 'all'], 
                        default='all', help='Source to use')
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    print(f"Getting real-time news for {ticker}")
    print(f"Source: {args.source}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_texts = []
    
    if args.source == 'marketaux' or args.source == 'all':
        news = get_marketaux_news(ticker)
        all_texts.extend([item["title"] for item in news])
        all_texts.extend([item["description"] for item in news if item.get("description")])
        
        if args.source == 'all':
            print("\n" + "-"*50 + "\n")
            time.sleep(1)  # Slight delay between requests
    
    if args.source == 'gnews' or args.source == 'all':
        news = get_gnews(ticker)
        all_texts.extend([item["title"] for item in news])
        
        if args.source == 'all':
            print("\n" + "-"*50 + "\n")
            time.sleep(1)  # Slight delay between requests
    
    if args.source == 'reddit' or args.source == 'all':
        posts = get_reddit_data(ticker)
        all_texts.extend([item["title"] for item in posts])
    
    if all_texts:
        print("\nAnalyzing sentiment based on gathered texts...")
        sentiment_results = analyze_sentiment(all_texts)
        print(f"\nSentiment Analysis for {ticker}:")
        print(json.dumps(sentiment_results, indent=2))
        
        # Save sentiment results
        with open(f"{ticker}_sentiment.json", "w") as f:
            json.dump(sentiment_results, f, indent=2)
        print(f"\nSentiment results saved to {ticker}_sentiment.json")
    else:
        print(f"\nNo texts were gathered for {ticker}, cannot perform sentiment analysis.")

if __name__ == "__main__":
    main()