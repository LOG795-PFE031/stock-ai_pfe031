import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import logging
import time
from typing import Tuple, List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from yfinance.exceptions import YFRateLimitError
from cachetools import TTLCache, cached
from ratelimit import limits, sleep_and_retry
from functools import lru_cache
import requests.exceptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
NEWS_CACHE = TTLCache(maxsize=100, ttl=300)  # Cache for 5 minutes
CALLS_PER_MINUTE = 1  # Reduced from 2 to be more conservative
ONE_MINUTE = 60

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=ONE_MINUTE)
@retry(
    stop=stop_after_attempt(5),  # Increased from 3 to 5 attempts
    wait=wait_exponential(multiplier=2, min=4, max=30),  # Increased wait times
    retry=retry_if_exception_type((YFRateLimitError, requests.exceptions.RequestException))
)
def get_todays_news_details(ticker: str) -> List[Dict[str, Any]]:
    """Get news details for a ticker with retry logic and rate limiting."""
    cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d_%H')}"
    
    # Check cache first
    if cache_key in NEWS_CACHE:
        logger.info(f"Returning cached news for {ticker}")
        return NEWS_CACHE[cache_key]
    
    try:
        # Add delay between requests
        time.sleep(2)  # Increased from 1 to 2 seconds
        
        # Initialize YFinance with custom config
        ticker_obj = yf.Ticker(ticker)
        
        # Try to get news
        news = ticker_obj.news
        
        if not news:
            logger.warning(f"No news found for ticker {ticker}")
            NEWS_CACHE[cache_key] = []
            return []
            
        # Cache the results
        NEWS_CACHE[cache_key] = news
        logger.info(f"Successfully fetched and cached news for {ticker}")
        return news
        
    except YFRateLimitError as e:
        logger.error(f"Rate limit hit for {ticker}: {str(e)}")
        # If we hit rate limit, try to return cached data if available
        if cache_key in NEWS_CACHE:
            logger.info(f"Returning stale cached data for {ticker} due to rate limit")
            return NEWS_CACHE[cache_key]
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching news for {ticker}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching news for {ticker}: {str(e)}")
        raise

def get_todays_news_urls(ticker: str) -> Tuple[List[str], str]:
    """Get today's news URLs for a given stock ticker with caching."""
    try:
        details = get_todays_news_details(ticker)
        if not details:
            logger.info(f"No news details found for {ticker}")
            return [], ticker

        # Extract unique URLs and filter out None/empty values
        urls = []
        for item in details:
            url = item.get('link')
            if url and isinstance(url, str) and url.startswith('http'):
                urls.append(url)
        
        urls = list(set(urls))  # Remove duplicates
        
        if not urls:
            logger.warning(f"No valid URLs found in news details for {ticker}")
            return [], ticker
            
        logger.info(f"Found {len(urls)} unique news articles for {ticker}")
        return urls, ticker
        
    except YFRateLimitError as e:
        logger.error(f"Rate limit exceeded for {ticker}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing news URLs for {ticker}: {str(e)}")
        raise

def fetch_and_save_todays_news(ticker: str) -> None:
    """Fetch and save today's news for a given stock ticker."""
    try:
        urls, _ = get_todays_news_urls(ticker)
        if not urls:
            logger.warning(f"No news URLs found for {ticker}")
            return
            
        # Process URLs and save news (implementation depends on your needs)
        logger.info(f"Successfully processed {len(urls)} news articles for {ticker}")
        
    except Exception as e:
        logger.error(f"Error fetching and saving news for {ticker}: {e}")
        raise

def clear_cache() -> None:
    """Clear all caches."""
    NEWS_CACHE.clear()
    get_todays_news_urls.cache_clear()
    logger.info("All caches cleared")

if __name__ == "__main__":
    # Check if ticker is provided as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python deepcrawler.py TICKER")
        print("Example: python deepcrawler.py AAPL")
        sys.exit(1)
    
    # Get ticker from command-line argument
    ticker = sys.argv[1].upper()
    fetch_and_save_todays_news(ticker)