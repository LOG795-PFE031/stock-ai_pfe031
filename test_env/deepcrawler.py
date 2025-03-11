import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

def get_todays_news_details(ticker):
    """
    Fetch and return a list of news article details for the given ticker from today.
    Each article is a dictionary with title, date, summary, displayTime, link, and source.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    news = yf.Ticker(ticker).news
    today_news = []
    
    for item in news:
        news_date = None
        # Extract date from 'displayTime', which may be in item or item's content
        if 'displayTime' in item:
            news_date = item['displayTime'].split('T')[0]
        elif 'content' in item and 'displayTime' in item['content']:
            news_date = item['content']['displayTime'].split('T')[0]
        
        if news_date == today:
            article = {}
            if 'content' in item:
                content = item['content']
                article['title'] = content.get('title', 'No Title')
                article['date'] = news_date
                article['summary'] = content.get('summary', '')
                article['displayTime'] = content.get('displayTime', '')
                # Extract URL from possible nested structures
                if 'clickThroughUrl' in content and content['clickThroughUrl'] and 'url' in content['clickThroughUrl']:
                    article['link'] = content['clickThroughUrl']['url']
                elif 'canonicalUrl' in content and content['canonicalUrl'] and 'url' in content['canonicalUrl']:
                    article['link'] = content['canonicalUrl']['url']
                else:
                    article['link'] = ''
                # Extract source if available
                if 'provider' in content and content['provider']:
                    article['source'] = content['provider'].get('displayName', 'Unknown')
                else:
                    article['source'] = 'Unknown'
            else:
                # Handle direct structure
                article['title'] = item.get('title', 'No Title')
                article['date'] = news_date
                article['summary'] = item.get('summary', '')
                article['displayTime'] = item.get('displayTime', '')
                article['link'] = item.get('link', '')
                article['source'] = 'Unknown'
            today_news.append(article)
    
    return today_news

def get_todays_news_urls(ticker):
    """
    Return a tuple containing a list of URLs for today's news articles and the ticker.
    """
    details = get_todays_news_details(ticker)
    urls = [article['link'] for article in details if article['link']]
    return (urls, ticker)

def fetch_and_save_todays_news(ticker):
    """
    Fetch today's news for the ticker, print details, and save to a CSV file.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    details = get_todays_news_details(ticker)
    
    if details:
        # Print news details
        print("\nToday's News")
        print(f"Found {len(details)} news items for today ({today})")
        for item in details:
            print(f"- {item['title']}")
            print(f"  Published: {item['displayTime']}")
            print(f"  Source: {item['source']}")
            if item['summary']:
                print(f"  Summary: {item['summary'][:100]}..." if len(item['summary']) > 100 else f"  Summary: {item['summary']}")
            print(f"  URL: {item['link']}")
            print("---")
        
        # Save to CSV
        news_dir = os.path.join("news", today)
        os.makedirs(news_dir, exist_ok=True)
        df = pd.DataFrame(details)
        csv_path = os.path.join(news_dir, f"{ticker}_today_news.csv")
        df.to_csv(csv_path, index=False)
        print(f"Today's news has been saved to {csv_path}")
    else:
        print("No news found for today.")

if __name__ == "__main__":
    # Check if ticker is provided as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python deepcrawler.py TICKER")
        print("Example: python deepcrawler.py AAPL")
        sys.exit(1)
    
    # Get ticker from command-line argument
    ticker = sys.argv[1].upper()
    fetch_and_save_todays_news(ticker)