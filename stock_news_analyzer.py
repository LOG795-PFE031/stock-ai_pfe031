#!/usr/bin/env python
# coding: utf-8

"""
Stock News Analyzer - Gathers news articles for a given stock ticker and performs sentiment analysis using FinBERT-tone.
"""

import argparse
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config
import concurrent.futures
from urllib.parse import urlparse
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import nltk
from datetime import datetime

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading additional NLTK data for newspaper...")
    nltk.download('punkt_tab', quiet=True)

class FinBERTSentimentAnalyzer:
    def __init__(self):
        """Initialize the FinBERT sentiment analyzer with pre-trained model and tokenizer."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_id = 0
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_id = -1
        else:
            self.device = torch.device("cpu")
            device_id = -1
        print(f"Using device: {self.device}")
        
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model.to(self.device)
        
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=device_id)
        self.label_map = {"neutral": 0, "positive": 1, "negative": 2}
    
    def batch_analyze(self, texts):
        """Analyze the sentiment of multiple texts."""
        results = self.pipeline(texts)
        processed_results = []
        for i, result in enumerate(results):
            sentiment = result['label'].lower()
            inputs = self.tokenizer(texts[i], return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            processed_results.append({
                "sentiment": sentiment,
                "confidence": result['score'],
                "scores": {
                    "positive": probs[1].item(),
                    "negative": probs[2].item(),
                    "neutral": probs[0].item()
                }
            })
        return processed_results

class StockNewsScraper:
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
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.47",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
        ]
        print("Initializing FinBERT sentiment analyzer...")
        self.analyzer = FinBERTSentimentAnalyzer()
        print("FinBERT analyzer ready.")
    
    def _get_company_name(self):
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
    
    def _get_random_user_agent(self):
        return random.choice(self.user_agents)
    
    def _get_google_news(self):
        search_query = f"{self.ticker} {self.company_name} stock news"
        search_url = f"https://news.google.com/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": self._get_random_user_agent()}
        try:
            response = requests.get(search_url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch Google News: status code {response.status_code}")
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            article_elements = soup.select("article")
            urls = []
            for article in article_elements:
                link = article.select_one("a[href]")
                if not link:
                    link = article.select_one("a")
                if link and link.has_attr("href"):
                    href = link["href"]
                    if href.startswith("./"):
                        url = f"https://news.google.com{href[1:]}"
                        urls.append(url)
                    elif href.startswith("/"):
                        url = f"https://news.google.com{href}"
                        urls.append(url)
                    elif href.startswith("http"):
                        urls.append(href)
            if not urls:
                all_links = soup.select("a.VDXfz")
                for link in all_links:
                    if link.has_attr("href"):
                        href = link["href"]
                        if href.startswith("./"):
                            url = f"https://news.google.com{href[1:]}"
                            urls.append(url)
            return urls[:self.max_articles]
        except Exception as e:
            print(f"Error fetching Google News: {e}")
            return []
    
    def _get_yahoo_finance(self):
        search_url = f"https://finance.yahoo.com/quote/{self.ticker}/news"
        headers = {"User-Agent": self._get_random_user_agent()}
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Failed to fetch Yahoo Finance: status code {response.status_code}")
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.select('li[data-test="stream-item"] a')
            urls = []
            for article in articles:
                if article.has_attr("href"):
                    href = article["href"]
                    if href.startswith("/"):
                        url = f"https://finance.yahoo.com{href}"
                    else:
                        url = href
                    urls.append(url)
            return urls[:self.max_articles]
        except Exception as e:
            print(f"Error fetching Yahoo Finance: {e}")
            return []
    
    def _get_seeking_alpha(self):
        search_url = f"https://seekingalpha.com/symbol/{self.ticker}"
        headers = {"User-Agent": self._get_random_user_agent()}
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Failed to fetch Seeking Alpha: status code {response.status_code}")
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            article_links = soup.select('a[data-test-id="post-list-item-title"]')
            if not article_links:
                article_links = soup.select('.title-link')
            urls = []
            for link in article_links:
                if link.has_attr("href"):
                    href = link["href"]
                    if href.startswith("/"):
                        url = f"https://seekingalpha.com{href}"
                    else:
                        url = href
                    urls.append(url)
            return urls[:self.max_articles]
        except Exception as e:
            print(f"Error fetching Seeking Alpha: {e}")
            return []
    
    def _get_market_watch(self):
        search_url = f"https://www.marketwatch.com/investing/stock/{self.ticker.lower()}"
        headers = {"User-Agent": self._get_random_user_agent()}
        try:
            response = requests.get(search_url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch MarketWatch: status code {response.status_code}")
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            article_links = soup.select('.article__content a.link')
            urls = []
            for link in article_links:
                if link.has_attr("href"):
                    href = link["href"]
                    if href.startswith("http"):
                        urls.append(href)
                    else:
                        url = f"https://www.marketwatch.com{href}"
                        urls.append(url)
            return urls[:self.max_articles]
        except Exception as e:
            print(f"Error fetching MarketWatch: {e}")
            return []
    
    def _get_business_insider(self):
        search_query = f"{self.ticker} {self.company_name} stock"
        search_url = f"https://www.businessinsider.com/s?q={search_query}"
        headers = {"User-Agent": self._get_random_user_agent()}
        try:
            response = requests.get(search_url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch Business Insider: status code {response.status_code}")
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            article_links = soup.select('h2.tout-title a')
            urls = []
            for link in article_links:
                if link.has_attr("href"):
                    href = link["href"]
                    if href.startswith("http"):
                        urls.append(href)
                    else:
                        url = f"https://www.businessinsider.com{href}"
                        urls.append(url)
            return urls[:self.max_articles]
        except Exception as e:
            print(f"Error fetching Business Insider: {e}")
            return []
    
    def _follow_redirect(self, url):
        try:
            headers = {"User-Agent": self._get_random_user_agent()}
            response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            if response.url:
                return response.url
        except Exception as e:
            print(f"Error following redirect: {e}")
        return url
    
    def _process_article(self, url):
        try:
            if "news.google.com" in url:
                url = self._follow_redirect(url)
            if not url or url.startswith("data:"):
                return None
            config = Config()
            config.browser_user_agent = self._get_random_user_agent()
            config.request_timeout = 20
            config.memoize_articles = False
            config.fetch_images = False
            config.follow_meta_refresh = True
            config.headers = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
                'TE': 'Trailers',
            }
            article = Article(url, config=config)
            try:
                article.download()
            except Exception as download_error:
                print(f"Download error for {url}: {download_error}")
                return None
            if article.html == "" or article.html is None:
                print(f"Empty HTML content for {url}")
                return None
            try:
                article.parse()
            except Exception as parse_error:
                print(f"Parsing error for {url}: {parse_error}")
                return None
            title = article.title
            text = article.text
            if not title or not text or len(text) < 150:
                print(f"Not enough content for {url}")
                return None
            domain = urlparse(url).netloc
            source = domain.replace('www.', '')
            if 'yahoo' in domain:
                source = "Yahoo Finance"
            elif 'marketwatch' in domain:
                source = "MarketWatch"
            elif 'seekingalpha' in domain:
                source = "Seeking Alpha"
            elif 'businessinsider' in domain:
                source = "Business Insider"
            ticker_mention_score = 0
            company_mention_score = 0
            if self.ticker.lower() in title.lower():
                ticker_mention_score += 5
            if self.company_name.lower() in title.lower():
                company_mention_score += 5
            ticker_mention_score += text.lower().count(self.ticker.lower())
            company_mention_score += text.lower().count(self.company_name.lower())
            relevance_score = ticker_mention_score + company_mention_score
            if relevance_score < 1:
                print(f"Low relevance for {url}")
                return None
            result = {
                "url": url,
                "title": title,
                "text": text,
                "source": source,
                "relevance_score": relevance_score
            }
            return result
        except Exception as e:
            print(f"Error processing article {url}: {e}")
            return None
    
    def get_news(self):
        print(f"Searching for news about {self.ticker} ({self.company_name})...")
        print("-" * 50)
        all_urls = []
        successful_sources = 0
        for i, source_method in enumerate(self.sources):
            source_name = source_method.__name__.replace('_get_', '')
            print(f"[{i+1}/{len(self.sources)}] Checking {source_name}...")
            try:
                source_urls = source_method()
                if source_urls:
                    all_urls.extend(source_urls)
                    successful_sources += 1
                    print(f"  ✓ Found {len(source_urls)} potential articles")
                else:
                    print(f"  ✗ No articles found")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
            time.sleep(random.uniform(1.0, 2.0))
        all_urls = list(set(all_urls))
        if not all_urls:
            print("\nNo articles found from any source.")
            return []
        print("\n" + "-" * 50)
        print(f"Found {len(all_urls)} potential articles from {successful_sources} sources")
        print("Starting content extraction with newspaper4k...")
        print("-" * 50)
        articles = []
        completed = 0
        successful = 0
        direct_urls = [url for url in all_urls if not "news.google.com" in url]
        google_urls = [url for url in all_urls if "news.google.com" in url]
        ordered_urls = direct_urls + google_urls
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {executor.submit(self._process_article, url): url for url in ordered_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                completed += 1
                try:
                    article = future.result()
                    if article:
                        articles.append(article)
                        successful += 1
                        domain = urlparse(url).netloc.replace('www.', '')
                        title_preview = article['title'][:40] + "..." if len(article['title']) > 40 else article['title']
                        print(f"[{completed}/{len(all_urls)}] ✓ {domain}: \"{title_preview}\"")
                    else:
                        print(f"[{completed}/{len(all_urls)}] ✗ Failed to extract from {url}")
                except Exception as e:
                    print(f"[{completed}/{len(all_urls)}] ✗ Error processing {url}: {e}")
                if len(articles) >= self.max_articles * 1.5:
                    break
        print("-" * 50)
        print(f"Successfully extracted {successful} articles out of {len(all_urls)} URLs")
        if not articles:
            print("No relevant articles found.")
            return []
        articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        articles = articles[:self.max_articles]
        print(f"Selected top {len(articles)} most relevant articles")
        if articles:
            texts = [article['text'] for article in articles]
            sentiment_results = self.analyzer.batch_analyze(texts)
            for article, sentiment in zip(articles, sentiment_results):
                article['sentiment_analysis'] = sentiment
        return articles
    
    def scrape_to_json(self, output_file=None):
        articles = self.get_news()
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
            print(f"Results saved to {output_file}")
        return result

def main():
    parser = argparse.ArgumentParser(description='Scrape news articles for a stock ticker and perform sentiment analysis using FinBERT-tone.')
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--output', '-o', help='Output file for JSON results')
    parser.add_argument('--articles', '-a', type=int, default=5, help='Maximum number of articles to return')
    parser.add_argument('--pretty', '-p', action='store_true', help='Pretty print the JSON output')
    args = parser.parse_args()
    scraper = StockNewsScraper(args.ticker, max_articles=args.articles)
    result = scraper.scrape_to_json(args.output)
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result))

if __name__ == "__main__":
    main()