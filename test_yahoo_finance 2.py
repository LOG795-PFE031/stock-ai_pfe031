#!/usr/bin/env python
# coding: utf-8

"""
Simple test script for Yahoo Finance webscraping.
"""

import yfinance as yf
import json

def get_yahoo_news(ticker):
    """Get latest news for a ticker from Yahoo Finance."""
    try:
        print(f"Fetching news for {ticker} using yfinance...")
        stock = yf.Ticker(ticker)
        
        # Get basic info
        print("\nBasic Info:")
        info = stock.info
        for key in ['shortName', 'longName', 'sector', 'industry', 'currentPrice', 'targetHighPrice', 'targetLowPrice']:
            if key in info:
                print(f"{key}: {info[key]}")
        
        # Get news
        print("\nLatest News:")
        news = stock.news
        
        results = []
        for i, item in enumerate(news[:5], 1):
            print(f"\n{i}. {item.get('title', 'No title')}")
            print(f"   Published: {item.get('providerPublishTime', 'Unknown')}")
            if 'link' in item:
                print(f"   Link: {item['link']}")
            
            if 'summary' in item:
                summary = item['summary']
                print(f"   Summary: {summary[:100]}..." if len(summary) > 100 else f"   Summary: {summary}")
            
            # Add to results
            results.append({
                "title": item.get('title', 'No title'),
                "summary": item.get('summary', 'No summary'),
                "published": item.get('providerPublishTime', 'Unknown'),
                "link": item.get('link', 'No link')
            })
        
        # Save results to file
        with open(f"{ticker}_yahoo_news.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {ticker}_yahoo_news.json")
        
        return True
    
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    ticker = "AAPL"  # Default ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    
    get_yahoo_news(ticker)