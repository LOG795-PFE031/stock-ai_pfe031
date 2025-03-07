#!/usr/bin/env python
# coding: utf-8

"""
Stock News Analyzer - A product-grade tool to scrape news articles for a stock ticker
and perform sentiment analysis using FinBERT-tone.
"""

import argparse
import json
import random
import time
from datetime import datetime
import logging
from tqdm import tqdm
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from newspaper import Article, Config
from urllib.parse import urlparse
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import nltk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        logging.info(f"Downloading NLTK {resource}...")
        nltk.download(resource, quiet=True)

class FinBERTSentimentAnalyzer:
    """Handles sentiment analysis using the FinBERT-tone model."""
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_id = 0
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_id = -1
        else:
            self.device = torch.device("cpu")
            device_id = -1
        logging.info(f"Using device: {self.device}")
        
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model.to(self.device)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=device_id)
    
    def batch_analyze(self, texts):
        """Analyze sentiment for a batch of texts."""
        try:
            results = self.pipeline(texts)
            processed_results = []
            for i, result in enumerate(results):
                sentiment = result['label'].lower()
                inputs = self.tokenizer(texts[i], return_tensors="pt", truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                processed_results.append({
                    "sentiment": sentiment,
                    "confidence": result['score'],
                    "scores": {"positive": probs[1].item(), "negative": probs[2].item(), "neutral": probs[0].item()}
                })
            return processed_results
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {e}")
            return []

class StockNewsScraper:
    """Scrapes news articles for a stock ticker and analyzes their sentiment."""
    def __init__(self, ticker, max_articles=5):
        self.ticker = ticker.upper()
        self.company_name = self._get_company_name()
        self.max_articles = max_articles
        self.sources = [
            self._get_google_news,
            self._get_yahoo_finance,
            self._get_seeking_alpha,
            self._get_market_watch,
            self._get_business_insider,
        ]
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/94.0.4606.81 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
        ]
        logging.info("Initializing FinBERT sentiment analyzer...")
        self.analyzer = FinBERTSentimentAnalyzer()
        logging.info("FinBERT analyzer ready.")

    def _get_company_name(self):
        """Retrieve company name from ticker (simple cache)."""
        common_tickers = {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google', 'GOOG': 'Google',
            'AMZN': 'Amazon', 'META': 'Meta', 'TSLA': 'Tesla', 'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan', 'V': 'Visa', 'WMT': 'Walmart'
        }
        return common_tickers.get(self.ticker, self.ticker)

    def _get_random_user_agent(self):
        """Return a random user agent for requests."""
        return random.choice(self.user_agents)

    async def _fetch_url(self, session, url, retries=3):
        """Fetch URL content with retries and exponential backoff."""
        headers = {"User-Agent": self._get_random_user_agent()}
        for attempt in range(retries):
            try:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        return await response.text()
                    logging.warning(f"Fetch failed for {url}: status {response.status}")
            except Exception as e:
                logging.error(f"Fetch error for {url}: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return None

    async def _get_google_news(self):
        """Fetch news from Google News."""
        search_query = f"{self.ticker} {self.company_name} stock news"
        url = f"https://news.google.com/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_url(session, url)
            if not html:
                return []
            soup = BeautifulSoup(html, "html.parser")
            urls = [f"https://news.google.com{link['href'][1:]}" for link in soup.select("a[href^='./']")][:self.max_articles]
            return urls

    async def _get_yahoo_finance(self):
        """Fetch news from Yahoo Finance."""
        url = f"https://finance.yahoo.com/quote/{self.ticker}/news"
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_url(session, url)
            if not html:
                return []
            soup = BeautifulSoup(html, "html.parser")
            urls = [f"https://finance.yahoo.com{a['href']}" if a['href'].startswith("/") else a['href']
                    for a in soup.select('li[data-test="stream-item"] a')][:self.max_articles]
            return urls

    async def _get_seeking_alpha(self):
        """Fetch news from Seeking Alpha."""
        url = f"https://seekingalpha.com/symbol/{self.ticker}"
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_url(session, url)
            if not html:
                return []
            soup = BeautifulSoup(html, "html.parser")
            urls = [f"https://seekingalpha.com{link['href']}" if link['href'].startswith("/") else link['href']
                    for link in soup.select('a[data-test-id="post-list-item-title"]')][:self.max_articles]
            return urls

    async def _get_market_watch(self):
        """Fetch news from MarketWatch."""
        url = f"https://www.marketwatch.com/investing/stock/{self.ticker.lower()}"
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_url(session, url)
            if not html:
                return []
            soup = BeautifulSoup(html, "html.parser")
            urls = [link['href'] if link['href'].startswith("http") else f"https://www.marketwatch.com{link['href']}"
                    for link in soup.select('.article__content a.link')][:self.max_articles]
            return urls

    async def _get_business_insider(self):
        """Fetch news from Business Insider."""
        search_query = f"{self.ticker} {self.company_name} stock"
        url = f"https://www.businessinsider.com/s?q={search_query}"
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_url(session, url)
            if not html:
                return []
            soup = BeautifulSoup(html, "html.parser")
            urls = [link['href'] if link['href'].startswith("http") else f"https://www.businessinsider.com{link['href']}"
                    for link in soup.select('h2.tout-title a')][:self.max_articles]
            return urls

    async def _follow_redirect(self, url):
        """Follow redirects to get the final URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(url, allow_redirects=True) as response:
                    return str(response.url) if response.url else url
        except Exception as e:
            logging.error(f"Redirect error for {url}: {e}")
            return url

    async def _process_article(self, url):
        """Process a single article: download, parse, and assess relevance."""
        if "news.google.com" in url:
            url = await self._follow_redirect(url)
        if not url or url.startswith("data:"):
            return None

        config = Config()
        config.browser_user_agent = self._get_random_user_agent()
        config.request_timeout = 20
        config.memoize_articles = False
        config.fetch_images = False
        config.headers = {'Accept': 'text/html', 'Referer': 'https://www.google.com/'}

        article = Article(url, config=config)
        try:
            article.download()
            if not article.html:
                logging.warning(f"Empty content for {url}")
                return None
            article.parse()
        except Exception as e:
            logging.error(f"Article processing error for {url}: {e}")
            return None

        title, text = article.title, article.text
        if not title or not text or len(text) < 150:
            logging.warning(f"Insufficient content for {url}")
            return None

        domain = urlparse(url).netloc.replace('www.', '')
        source = {
            'yahoo': 'Yahoo Finance', 'marketwatch': 'MarketWatch',
            'seekingalpha': 'Seeking Alpha', 'businessinsider': 'Business Insider'
        }.get(domain.split('.')[0], domain)

        relevance_score = (text.lower().count(self.ticker.lower()) + 
                           text.lower().count(self.company_name.lower()) +
                           (5 if self.ticker.lower() in title.lower() else 0) +
                           (5 if self.company_name.lower() in title.lower() else 0))
        if relevance_score < 1:
            logging.warning(f"Low relevance for {url}")
            return None

        return {"url": url, "title": title, "text": text, "source": source, "relevance_score": relevance_score}

    async def get_news(self):
        """Fetch and process news articles from multiple sources."""
        logging.info(f"Searching news for {self.ticker} ({self.company_name})...")
        tasks = [source() for source in self.sources]
        results = await asyncio.gather(*tasks)
        all_urls = list(set(url for sublist in results if sublist for url in sublist))
        
        if not all_urls:
            logging.info("No articles found.")
            return []

        logging.info(f"Found {len(all_urls)} unique URLs from {sum(1 for r in results if r)} sources")
        articles = []
        with tqdm(total=min(len(all_urls), self.max_articles * 2), desc="Processing articles") as pbar:
            for url in all_urls:
                article = await self._process_article(url)
                if article:
                    articles.append(article)
                pbar.update(1)
                if len(articles) >= self.max_articles * 1.5:
                    break
                await asyncio.sleep(0.5)  # Rate limiting

        articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        articles = articles[:self.max_articles]
        logging.info(f"Selected {len(articles)} most relevant articles")

        if articles:
            texts = [a['text'] for a in articles]
            sentiments = self.analyzer.batch_analyze(texts)
            for article, sentiment in zip(articles, sentiments):
                article['sentiment_analysis'] = sentiment
        return articles

    async def scrape_to_json(self, output_file=None):
        """Scrape news and save/output results as JSON."""
        articles = await self.get_news()
        result = {
            "ticker": self.ticker,
            "company": self.company_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "articles": articles,
            "total_articles": len(articles)
        }
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logging.info(f"Results saved to {output_file}")
        return result

def main():
    """Parse arguments and run the scraper."""
    parser = argparse.ArgumentParser(description="Scrape and analyze stock news with FinBERT-tone.")
    parser.add_argument('ticker', help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument('--output', '-o', help="Output JSON file")
    parser.add_argument('--articles', '-a', type=int, default=5, help="Max articles to fetch")
    parser.add_argument('--pretty', '-p', action='store_true', help="Pretty-print JSON output")
    args = parser.parse_args()

    scraper = StockNewsScraper(args.ticker, args.articles)
    result = asyncio.run(scraper.scrape_to_json(args.output))
    
    print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))

if __name__ == "__main__":
    main()