#!/usr/bin/env python
# coding: utf-8

"""
Direct webscraping test for financial news.
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import argparse

def get_user_agent():
    """Return a common user agent string."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

def scrape_market_watch(ticker):
    """Scrape MarketWatch for news related to a ticker."""
    print(f"Scraping MarketWatch for {ticker}...")
    
    url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
    
    try:
        headers = get_user_agent()
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Got status code {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Find news headlines in the Latest News section
        headlines = soup.select('.article__headline')
        print(f"Found {len(headlines)} headlines")
        
        for headline in headlines:
            text = headline.text.strip()
            if text and len(text) > 10:
                results.append(text)
                print(f"- {text}")
        
        # Find other related headlines
        other_headlines = soup.select('.link')
        print(f"Found {len(other_headlines)} other links")
        
        for headline in other_headlines:
            text = headline.text.strip()
            if text and len(text) > 10 and text not in results:
                results.append(text)
                print(f"- {text}")
        
        # Save results to file
        with open(f"{ticker}_marketwatch.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {ticker}_marketwatch.json")
        
        return results
    
    except Exception as e:
        print(f"Error scraping MarketWatch: {e}")
        return []

def scrape_seeking_alpha(ticker):
    """Scrape Seeking Alpha for news related to a ticker."""
    print(f"Scraping Seeking Alpha for {ticker}...")
    
    url = f"https://seekingalpha.com/symbol/{ticker}"
    
    try:
        headers = get_user_agent()
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Got status code {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Print the page title to check if we're on the right page
        title = soup.find('title')
        if title:
            print(f"Page title: {title.text}")
        
        # Find news headlines
        headlines = soup.select('a[sasq-type="ShortTitle"]')
        if not headlines:
            headlines = soup.select('a.sasq-189')  # Alternative selector
        
        print(f"Found {len(headlines)} headlines")
        
        for headline in headlines:
            text = headline.text.strip()
            if text and len(text) > 10:
                results.append(text)
                print(f"- {text}")
        
        # Try to get some other headlines
        other_headlines = soup.select('div.title')
        print(f"Found {len(other_headlines)} other headlines")
        
        for headline in other_headlines:
            text = headline.text.strip()
            if text and len(text) > 10 and text not in results:
                results.append(text)
                print(f"- {text}")
        
        # Save results to file
        with open(f"{ticker}_seekingalpha.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {ticker}_seekingalpha.json")
        
        return results
    
    except Exception as e:
        print(f"Error scraping Seeking Alpha: {e}")
        return []

def scrape_ft(ticker):
    """Scrape Financial Times for news related to a ticker."""
    print(f"Scraping Financial Times for {ticker}...")
    
    url = f"https://www.ft.com/stream/f914b887-4f95-47fe-b166-5152ae87e64a"  # Markets section
    
    try:
        headers = get_user_agent()
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Got status code {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Find headlines
        headlines = soup.select('a.js-teaser-heading-link')
        print(f"Found {len(headlines)} headlines")
        
        for headline in headlines:
            text = headline.text.strip()
            if text and len(text) > 10:
                results.append(text)
                print(f"- {text}")
        
        # Save results to file
        with open(f"{ticker}_ft.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {ticker}_ft.json")
        
        return results
    
    except Exception as e:
        print(f"Error scraping Financial Times: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Direct test of webscraping financial news.')
    parser.add_argument('ticker', type=str, help='Stock ticker to scrape news for', nargs='?', default='AAPL')
    parser.add_argument('--source', '-s', type=str, choices=['marketwatch', 'seekingalpha', 'ft', 'all'], 
                        default='all', help='Source to scrape')
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    print(f"Running direct webscraping test for {ticker}")
    print(f"Source: {args.source}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.source == 'marketwatch' or args.source == 'all':
        scrape_market_watch(ticker)
        if args.source == 'all':
            print("\n" + "-"*50 + "\n")
            time.sleep(1)  # Slight delay between requests
    
    if args.source == 'seekingalpha' or args.source == 'all':
        scrape_seeking_alpha(ticker)
        if args.source == 'all':
            print("\n" + "-"*50 + "\n")
            time.sleep(1)  # Slight delay between requests
    
    if args.source == 'ft' or args.source == 'all':
        scrape_ft(ticker)

if __name__ == "__main__":
    main()