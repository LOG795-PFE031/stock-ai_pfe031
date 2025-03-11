import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
from typing import List, Dict
from datetime import datetime
import re
from newspaper import Article, Config
from deepcrawler import get_todays_news_urls

async def crawl_and_extract(url: str, ticker: str) -> Dict:
    """
    Crawl a single URL and extract title, date, and content using newspaper3k.
    
    Args:
        url (str): The URL to crawl.
        ticker (str): The stock ticker associated with the news.
    
    Returns:
        Dict: A dictionary containing the URL, title, date, content, or error message.
    """
    try:
        # Configure newspaper with longer timeout and browser user-agent
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        config.request_timeout = 30
        
        # Use newspaper3k to extract article
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        # Parse the publication date
        publish_date = article.publish_date
        
        # If newspaper couldn't extract a date, use current date
        if not publish_date:
            publish_date = datetime.now()
            
        # Format date as string if available
        if publish_date:
            date_str = publish_date.strftime("%a, %B %d, %Y at %I:%M %p")
        else:
            date_str = "Unknown Date"
        
        return {
            "url": url,
            "title": article.title,
            "ticker": ticker,
            "content": article.text,
            "date": date_str
        }
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return {}

async def main(urls: List[str], ticker: str):
    """
    Main function to crawl multiple URLs and save results to a JSON file.
    
    Args:
        urls (List[str]): List of URLs received from deepcrawler.py.
        ticker (str): The stock ticker.
    """
    # Filter URLs to include only complete URLs
    valid_urls = [url for url in urls if url.startswith('http')]
    print(f"Found {len(valid_urls)} valid URLs to crawl.")
    
    if not valid_urls:
        print("No valid URLs found to crawl.")
        return
        
    # Crawl all URLs concurrently
    tasks = [crawl_and_extract(url, ticker) for url in valid_urls]
    results = await asyncio.gather(*tasks)
    
    # Filter out empty results
    results = [r for r in results if r]
    
    # Save results to a JSON file
    with open("articles.json", "w") as f:
        json.dump(results, f, indent=4)
        print(f"Saved {len(results)} articles to 'articles.json'.")

if __name__ == "__main__":
    # Read ticker from the user on CLI
    ticker = input("Enter the ticker: ")
    
    urls, ticker = get_todays_news_urls(ticker)
    asyncio.run(main(urls, ticker))