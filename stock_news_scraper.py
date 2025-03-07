#!/usr/bin/env python
# coding: utf-8

"""
Stock News Scraper - Gathers news articles for a given stock ticker.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import argparse
import random
import re
from datetime import datetime
from urllib.parse import urlparse, urljoin

class StockNewsScraper:
    """Scrapes news articles for a given stock ticker from multiple sources."""
    
    def __init__(self, ticker, max_sources=6, max_retries=3, timeout=10):
        """Initialize the scraper with the given ticker."""
        self.ticker = ticker.upper()
        self.company_name = self._get_company_name()
        self.max_sources = max_sources
        self.max_retries = max_retries
        self.timeout = timeout
        self.sources = []
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        ]
    
    def _get_company_name(self):
        """Get the company name for the ticker. Could be expanded with a more extensive lookup."""
        # This is a simplified version - in production you might use a more complete API
        common_tickers = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'GOOG': 'Google',
            'AMZN': 'Amazon',
            'META': 'Meta',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan',
            'V': 'Visa',
            'WMT': 'Walmart'
        }
        return common_tickers.get(self.ticker, self.ticker)
    
    def get_headers(self):
        """Return a random user agent string."""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
    
    def _clean_text(self, text):
        """Clean and normalize article text."""
        if not text:
            return ""
        
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common advertising phrases
        text = re.sub(r'(advertisement|sponsored content|paid content|subscribe now|sign up|paywall|premium content)', '', text, flags=re.IGNORECASE)
        
        # Remove copyright notices
        text = re.sub(r'copyright.*?reserved\.?', '', text, flags=re.IGNORECASE)
        
        # Remove "By Author Name" patterns
        text = re.sub(r'by\s+[A-Z][a-z]+\s+[A-Z][a-z]+', '', text)
        
        return text.strip()
    
    def _extract_article_text(self, url):
        """Extract the main text and metadata from an article URL."""
        for _ in range(self.max_retries):
            try:
                headers = self.get_headers()
                response = requests.get(url, headers=headers, timeout=self.timeout)
                
                if response.status_code != 200:
                    print(f"Failed to access {url}: Status code {response.status_code}")
                    time.sleep(1)
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get potential date from metadata first
                published_date = ""
                
                # Try to find date in meta tags
                meta_date_tags = [
                    soup.find('meta', {'property': 'article:published_time'}),
                    soup.find('meta', {'name': 'date'}),
                    soup.find('meta', {'name': 'publication_date'}),
                    soup.find('meta', {'name': 'article.published'}),
                    soup.find('meta', {'itemprop': 'datePublished'})
                ]
                
                for meta_tag in meta_date_tags:
                    if meta_tag and meta_tag.get('content'):
                        date_str = meta_tag.get('content')
                        # Extract just the date part if it's in ISO format
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
                        if date_match:
                            published_date = date_match.group(1)
                            break
                
                # If we still don't have a date, try looking for time elements
                if not published_date:
                    time_elements = soup.find_all('time')
                    for time_elem in time_elements:
                        if time_elem.get('datetime'):
                            date_str = time_elem.get('datetime')
                            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
                            if date_match:
                                published_date = date_match.group(1)
                                break
                
                # Remove unwanted elements
                for unwanted in soup.select('script, style, nav, footer, .ad, .advertisement, .banner'):
                    unwanted.decompose()
                
                # Try various common article content selectors
                article_selectors = [
                    'article', '.article', '.article-body', '.article-content', 
                    '.story-body', '.story-content', '.content-body', '.entry-content',
                    '.post-content', '#article-body', '.main-content', '.body'
                ]
                
                article_text = ""
                for selector in article_selectors:
                    content = soup.select_one(selector)
                    if content:
                        # Get all paragraphs
                        paragraphs = content.find_all('p')
                        if paragraphs:
                            article_text = ' '.join([p.get_text() for p in paragraphs])
                            break
                
                # Fallback: if no article content found with selectors, try getting all paragraphs
                if not article_text:
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        # Filter out very short paragraphs (likely not main content)
                        article_text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
                
                # If we still don't have a date, try to extract from the article text
                if not published_date:
                    # Look for common date patterns in the article text
                    date_patterns = [
                        r'(\d{4}-\d{2}-\d{2})',  # ISO format
                        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}',  # Month Day, Year
                        r'(\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})'  # Day Month Year
                    ]
                    
                    for pattern in date_patterns:
                        date_match = re.search(pattern, article_text)
                        if date_match:
                            published_date = date_match.group(0)
                            break
                
                return self._clean_text(article_text), published_date
                
            except Exception as e:
                print(f"Error extracting article text from {url}: {e}")
                time.sleep(1)
        
        return "", ""
    
    def _extract_article_metadata(self, url, content):
        """Extract metadata from the article."""
        domain = urlparse(url).netloc
        title = ""
        published_date = ""
        
        # Try to extract title and date if we have content
        if content:
            # Basic extraction - this could be improved with more specific selectors per site
            title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1)
            
            # Look for common date patterns in the content
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',  # ISO format
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}',  # Month Day, Year
                r'(\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})'  # Day Month Year
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, content)
                if date_match:
                    published_date = date_match.group(0)
                    break
        
        return {
            "source": domain,
            "title": self._clean_text(title),
            "published_date": published_date
        }
    
    def scrape_yahoo_finance(self):
        """Scrape Yahoo Finance for news related to a ticker."""
        print(f"Scraping Yahoo Finance for {self.ticker}...")
        
        url = f"https://finance.yahoo.com/quote/{self.ticker}/news"
        articles = []
        
        try:
            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            article_elements = soup.select('li[data-test="stream-item"]')
            
            for article in article_elements[:3]:  # Limit to 3 articles
                link_element = article.select_one('a')
                
                if link_element and link_element.has_attr('href'):
                    article_url = link_element['href']
                    
                    # Handle relative URLs
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://finance.yahoo.com', article_url)
                    
                    title_element = article.select_one('h3')
                    title = title_element.text.strip() if title_element else ""
                    
                    # Extract article text and date
                    article_text, published_date = self._extract_article_text(article_url)
                    
                    if article_text:
                        articles.append({
                            "url": article_url,
                            "title": title,
                            "source": "Yahoo Finance",
                            "text": article_text,
                            "published_date": published_date
                        })
                
                # If we have enough articles, break
                if len(articles) >= 2:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error scraping Yahoo Finance: {e}")
            return []
    
    def scrape_market_watch(self):
        """Scrape MarketWatch for news related to a ticker."""
        print(f"Scraping MarketWatch for {self.ticker}...")
        
        url = f"https://www.marketwatch.com/investing/stock/{self.ticker.lower()}"
        articles = []
        
        try:
            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            article_elements = soup.select('.article__content')
            
            for article in article_elements[:3]:  # Limit to 3 articles
                link_element = article.select_one('a.link')
                
                if link_element and link_element.has_attr('href'):
                    article_url = link_element['href']
                    
                    # Handle relative URLs
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://www.marketwatch.com', article_url)
                    
                    title_element = article.select_one('.article__headline')
                    title = title_element.text.strip() if title_element else ""
                    # Clean title - remove newlines and excess whitespace
                    title = re.sub(r'\s+', ' ', title).strip()
                    
                    # Extract article text and date
                    article_text, published_date = self._extract_article_text(article_url)
                    
                    if article_text:
                        articles.append({
                            "url": article_url,
                            "title": title,
                            "source": "MarketWatch",
                            "text": article_text,
                            "published_date": published_date
                        })
                
                # If we have enough articles, break
                if len(articles) >= 2:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error scraping MarketWatch: {e}")
            return []
    
    def scrape_seeking_alpha(self):
        """Scrape Seeking Alpha for news related to a ticker."""
        print(f"Scraping Seeking Alpha for {self.ticker}...")
        
        url = f"https://seekingalpha.com/symbol/{self.ticker}"
        articles = []
        
        try:
            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles - note that Seeking Alpha might use dynamic content loading
            article_links = soup.select('a[data-test-id="post-list-item-title"]')
            if not article_links:
                article_links = soup.select('a.sasq-189')  # Alternative selector
            
            for link in article_links[:3]:  # Limit to 3 articles
                if link.has_attr('href'):
                    article_url = link['href']
                    
                    # Handle relative URLs
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://seekingalpha.com', article_url)
                    
                    title = link.text.strip()
                    
                    # Extract article text and date
                    article_text, published_date = self._extract_article_text(article_url)
                    
                    if article_text:
                        articles.append({
                            "url": article_url,
                            "title": title,
                            "source": "Seeking Alpha",
                            "text": article_text,
                            "published_date": published_date
                        })
                
                # If we have enough articles, break
                if len(articles) >= 2:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error scraping Seeking Alpha: {e}")
            return []
    
    def scrape_reuters(self):
        """Scrape Reuters for news related to a ticker."""
        print(f"Scraping Reuters for {self.ticker}...")
        
        # Try to search for the company name rather than just the ticker
        search_term = self.company_name if self.company_name != self.ticker else self.ticker
        url = f"https://www.reuters.com/search/news?blob={search_term}"
        articles = []
        
        try:
            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            article_elements = soup.select('.search-result-content')
            
            for article in article_elements[:3]:  # Limit to 3 articles
                link_element = article.select_one('a.text-size--medium')
                
                if link_element and link_element.has_attr('href'):
                    article_url = link_element['href']
                    
                    # Handle relative URLs
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://www.reuters.com', article_url)
                    
                    title = link_element.text.strip()
                    
                    # Extract article text and date
                    article_text, published_date = self._extract_article_text(article_url)
                    
                    if article_text:
                        articles.append({
                            "url": article_url,
                            "title": title,
                            "source": "Reuters",
                            "text": article_text,
                            "published_date": published_date
                        })
                
                # If we have enough articles, break
                if len(articles) >= 2:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error scraping Reuters: {e}")
            return []
    
    def scrape_google_news(self):
        """Scrape Google News for news related to a ticker."""
        print(f"Scraping Google News for {self.ticker}...")
        
        # Create a search term that includes both ticker and company name
        search_term = f"{self.ticker}+{self.company_name}+stock+news"
        url = f"https://news.google.com/search?q={search_term}"
        articles = []
        
        try:
            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            article_elements = soup.select('article')
            
            for article in article_elements[:3]:  # Limit to 3 articles
                link_element = article.select_one('a')
                
                if link_element and link_element.has_attr('href'):
                    article_url = link_element['href']
                    
                    # Google News uses a special URL format, so we need to extract the actual URL
                    if article_url.startswith('./'):
                        article_url = urljoin('https://news.google.com', article_url.replace('./', '/'))
                    
                    title_element = article.select_one('h3, h4')
                    title = title_element.text.strip() if title_element else ""
                    
                    # For Google News, we need to find the actual article link
                    # This might require an additional request to follow the redirect
                    try:
                        redirect_response = requests.head(article_url, headers=headers, timeout=self.timeout, allow_redirects=True)
                        if redirect_response.url:
                            article_url = redirect_response.url
                    except Exception:
                        # If we can't follow the redirect, just use the original URL
                        pass
                    
                    # Extract article text and date
                    article_text, published_date = self._extract_article_text(article_url)
                    
                    if article_text:
                        articles.append({
                            "url": article_url,
                            "title": title,
                            "source": "Google News",
                            "text": article_text,
                            "published_date": published_date
                        })
                
                # If we have enough articles, break
                if len(articles) >= 2:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error scraping Google News: {e}")
            return []
    
    def scrape_financial_times(self):
        """Scrape Financial Times for news related to a ticker."""
        print(f"Scraping Financial Times for {self.ticker}...")
        
        # Search for the company name in the markets section
        search_term = self.company_name if self.company_name != self.ticker else self.ticker
        url = f"https://www.ft.com/search?q={search_term}&sort=date"
        articles = []
        
        try:
            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            article_elements = soup.select('.o-teaser')
            
            for article in article_elements[:3]:  # Limit to 3 articles
                link_element = article.select_one('a.js-teaser-heading-link')
                
                if link_element and link_element.has_attr('href'):
                    article_url = link_element['href']
                    
                    # Handle relative URLs
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://www.ft.com', article_url)
                    
                    title = link_element.text.strip()
                    
                    # FT has a paywall - we might not get the full content
                    article_text, published_date = self._extract_article_text(article_url)
                    
                    if article_text:
                        articles.append({
                            "url": article_url,
                            "title": title,
                            "source": "Financial Times",
                            "text": article_text,
                            "published_date": published_date
                        })
                
                # If we have enough articles, break
                if len(articles) >= 2:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error scraping Financial Times: {e}")
            return []
    
    def scrape_barrons(self):
        """Scrape Barron's for news related to a ticker."""
        print(f"Scraping Barron's for {self.ticker}...")
        
        url = f"https://www.barrons.com/search?keyword={self.ticker}&sort=date"
        articles = []
        
        try:
            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            article_elements = soup.select('.SearchResults-articleBody')
            
            for article in article_elements[:3]:  # Limit to 3 articles
                link_element = article.select_one('a')
                
                if link_element and link_element.has_attr('href'):
                    article_url = link_element['href']
                    
                    # Handle relative URLs
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://www.barrons.com', article_url)
                    
                    title_element = article.select_one('.SearchResults-headline')
                    title = title_element.text.strip() if title_element else ""
                    
                    # Extract article text and date
                    article_text, published_date = self._extract_article_text(article_url)
                    
                    if article_text:
                        articles.append({
                            "url": article_url,
                            "title": title,
                            "source": "Barron's",
                            "text": article_text,
                            "published_date": published_date
                        })
                
                # If we have enough articles, break
                if len(articles) >= 2:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error scraping Barron's: {e}")
            return []
    
    def scrape_cnbc(self):
        """Scrape CNBC for news related to a ticker."""
        print(f"Scraping CNBC for {self.ticker}...")
        
        # Try to search for both ticker and company name
        search_term = f"{self.ticker}+{self.company_name}+stock"
        url = f"https://www.cnbc.com/search/?query={search_term}&qsearchterm={self.ticker}"
        articles = []
        
        try:
            headers = self.get_headers()
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Error: Got status code {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            article_elements = soup.select('.SearchResult-searchResult')
            
            for article in article_elements[:3]:  # Limit to 3 articles
                link_element = article.select_one('a')
                
                if link_element and link_element.has_attr('href'):
                    article_url = link_element['href']
                    
                    title_element = article.select_one('.SearchResult-headline')
                    title = title_element.text.strip() if title_element else ""
                    
                    # Extract article text and date
                    article_text, published_date = self._extract_article_text(article_url)
                    
                    if article_text:
                        articles.append({
                            "url": article_url,
                            "title": title,
                            "source": "CNBC",
                            "text": article_text,
                            "published_date": published_date
                        })
                
                # If we have enough articles, break
                if len(articles) >= 2:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error scraping CNBC: {e}")
            return []
    
    def scrape_all_sources(self):
        """Scrape all sources for news related to a ticker."""
        all_sources = [
            self.scrape_yahoo_finance,
            self.scrape_market_watch,
            self.scrape_seeking_alpha,
            self.scrape_reuters,
            self.scrape_google_news,
            self.scrape_financial_times,
            self.scrape_cnbc,
            self.scrape_barrons
        ]
        
        # Shuffle the sources to randomize which ones we try first
        random.shuffle(all_sources)
        
        results = []
        sources_tried = 0
        
        # Try sources until we have enough articles or have tried all sources
        for source_scraper in all_sources:
            if len(results) >= self.max_sources:
                break
            
            articles = source_scraper()
            results.extend(articles)
            sources_tried += 1
            
            # Add a small delay between sources to avoid being blocked
            time.sleep(random.uniform(1, 2))
        
        # Check for relevance by looking for ticker or company name
        relevant_articles = []
        for article in results:
            article_text = article.get('text', '')
            article_title = article.get('title', '')
            
            # Check if ticker or company name is in title or text
            if (self.ticker.lower() in article_text.lower() or 
                self.ticker.lower() in article_title.lower() or
                self.company_name.lower() in article_text.lower() or
                self.company_name.lower() in article_title.lower()):
                
                # Add relevance score
                relevance = 0
                if self.ticker.lower() in article_title.lower():
                    relevance += 10
                if self.company_name.lower() in article_title.lower():
                    relevance += 8
                if self.ticker.lower() in article_text.lower():
                    relevance += 5
                if self.company_name.lower() in article_text.lower():
                    relevance += 3
                
                article['relevance_score'] = relevance
                relevant_articles.append(article)
            
        # Sort by relevance if we have relevant articles
        if relevant_articles:
            relevant_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            results = relevant_articles[:self.max_sources]
        else:
            # If no relevant articles found, just return what we have
            results = results[:self.max_sources]
        
        print(f"\nScraped {len(results)} articles from {sources_tried} sources")
        return results
    
    def scrape_to_json(self, output_file=None):
        """Scrape news articles and return as JSON."""
        articles = self.scrape_all_sources()
        
        # Create the result JSON
        result = {
            "ticker": self.ticker,
            "company": self.company_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "articles": articles,
            "total_articles": len(articles)
        }
        
        # If an output file is specified, write the JSON to it
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        
        return result

def main():
    """Main function to run the scraper from command line."""
    parser = argparse.ArgumentParser(description='Scrape news articles for a given stock ticker.')
    parser.add_argument('ticker', type=str, help='Stock ticker to scrape news for')
    parser.add_argument('--output', '-o', type=str, help='Output file for JSON results')
    parser.add_argument('--sources', '-s', type=int, default=6, help='Maximum number of sources to scrape')
    parser.add_argument('--pretty', '-p', action='store_true', help='Pretty print the JSON output')
    args = parser.parse_args()
    
    # Create the scraper
    scraper = StockNewsScraper(args.ticker, max_sources=args.sources)
    
    # Scrape the news
    result = scraper.scrape_to_json(args.output)
    
    # Print the result
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result))

if __name__ == "__main__":
    main()