#!/usr/bin/env python
# coding: utf-8

"""
Stock News Analyzer API - Scrapes news articles for a stock ticker,
performs sentiment analysis using FinBERT-tone, and publishes to RabbitMQ.
"""

import json
import random
import os
import sys
from datetime import datetime
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from newspaper import Article, Config
from urllib.parse import urlparse
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import nltk
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields

sys.path.append(os.path.dirname(__file__))
from news_publisher import NewsPublisher

# Logging setup
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'news_analyzer.log'))
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        logger.info(f"Downloading NLTK {resource}...")
        nltk.download(resource, quiet=True)

# Flask and API setup
app = Flask(__name__)
api = Api(app, version='1.0',
          title='Stock News Analyzer API',
          description='API for analyzing news sentiment for stocks')

ns = api.namespace('api', description='Stock News Analysis Operations')

# API models
stock_model = api.model('StockModel', {
    'ticker': fields.String(required=True, description='Stock ticker (e.g., AAPL)'),
    'articles': fields.Integer(default=5, description='Number of articles to analyze')
})

sentiment_model = api.model('SentimentModel', {
    'sentiment': fields.String(description='Sentiment (positive, negative, neutral)'),
    'confidence': fields.Float(description='Confidence score'),
    'scores': fields.Raw(description='Detailed sentiment scores')
})

article_model = api.model('ArticleModel', {
    'title': fields.String(description='Article title'),
    'source': fields.String(description='News source'),
    'url': fields.String(description='Article URL'),
    'text': fields.String(description='Article text'),
    'sentiment_analysis': fields.Nested(sentiment_model, description='Sentiment results'),
    'published_at': fields.String(description='Publication date')
})

response_model = api.model('ResponseModel', {
    'ticker': fields.String(description='Stock ticker'),
    'company': fields.String(description='Company name'),
    'timestamp': fields.String(description='Analysis timestamp'),
    'total_articles': fields.Integer(description='Number of articles analyzed'),
    'articles': fields.List(fields.Nested(article_model), description='Analyzed articles')
})

class FinBERTSentimentAnalyzer:
    """Handles sentiment analysis using FinBERT-tone with GPU support."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FinBERTSentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Device selection for GPU/CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_id = 0
            logger.info("CUDA available, using GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_id = -1
            logger.info("MPS available, using Apple Silicon GPU")
        else:
            self.device = torch.device("cpu")
            device_id = -1
            logger.info("No GPU available, using CPU")

        logger.info("Loading FinBERT model...")
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model.to(self.device)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=device_id)
        logger.info("FinBERT model loaded successfully")
        self._initialized = True

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
            logger.error(f"Sentiment analysis failed: {e}")
            return []

class StockNewsScraper:
    """Scrapes news articles and analyzes sentiment."""
    def __init__(self, ticker, max_articles=5, publish_to_rabbitmq=True):
        self.ticker = ticker.upper()
        self.company_name = self._get_company_name()
        self.max_articles = max_articles
        self.publish_to_rabbitmq = publish_to_rabbitmq
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
        self.analyzer = FinBERTSentimentAnalyzer()
        if self.publish_to_rabbitmq:
            self.publisher = NewsPublisher()

    def _get_company_name(self):
        """Simple ticker-to-company mapping."""
        common_tickers = {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google', 'GOOG': 'Google',
            'AMZN': 'Amazon', 'META': 'Meta', 'TSLA': 'Tesla', 'NVDA': 'NVIDIA'
        }
        return common_tickers.get(self.ticker, self.ticker)

    def _get_random_user_agent(self):
        """Return a random user agent."""
        return random.choice(self.user_agents)

    async def _fetch_url(self, session, url, retries=3):
        """Fetch URL content with retries."""
        headers = {"User-Agent": self._get_random_user_agent()}
        for attempt in range(retries):
            try:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        return await response.text()
                    logger.warning(f"Fetch failed for {url}: status {response.status}")
            except Exception as e:
                logger.error(f"Fetch error for {url}: {e}")
            await asyncio.sleep(2 ** attempt)
        return None

    async def _get_google_news(self):
        url = f"https://news.google.com/search?q={self.ticker}+{self.company_name}+stock+news&hl=en-US"
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_url(session, url)
            if not html:
                return []
            soup = BeautifulSoup(html, "html.parser")
            urls = [f"https://news.google.com{link['href'][1:]}" for link in soup.select("a[href^='./']")][:self.max_articles]
            return urls

    async def _get_yahoo_finance(self):
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
        url = f"https://www.businessinsider.com/s?q={self.ticker}+{self.company_name}+stock"
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
            logger.error(f"Redirect error for {url}: {e}")
            return url

    async def _process_article(self, url):
        """Process an article and extract relevant data."""
        if "news.google.com" in url:
            url = await self._follow_redirect(url)
        if not url or url.startswith("data:"):
            return None

        config = Config()
        config.browser_user_agent = self._get_random_user_agent()
        config.request_timeout = 20
        config.memoize_articles = False
        config.fetch_images = False

        article = Article(url, config=config)
        try:
            article.download()
            if not article.html:
                logger.warning(f"Empty content for {url}")
                return None
            article.parse()
            article.nlp()
        except Exception as e:
            logger.error(f"Article processing error for {url}: {e}")
            return None

        title, text, publish_date = article.title, article.text, article.publish_date
        if not title or not text or len(text) < 150:
            logger.warning(f"Insufficient content for {url}")
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
            logger.warning(f"Low relevance for {url}")
            return None

        return {
            "url": url,
            "title": title,
            "text": text,
            "source": source,
            "relevance_score": relevance_score,
            "publish_date": publish_date
        }

    async def get_news(self):
        """Fetch and process news articles."""
        logger.info(f"Searching news for {self.ticker} ({self.company_name})...")
        tasks = [source() for source in self.sources]
        results = await asyncio.gather(*tasks)
        all_urls = list(set(url for sublist in results if sublist for url in sublist))

        if not all_urls:
            logger.info("No articles found.")
            return []

        logger.info(f"Found {len(all_urls)} unique URLs")
        articles = []
        for url in all_urls:
            article = await self._process_article(url)
            if article:
                articles.append(article)
            if len(articles) >= self.max_articles * 1.5:
                break
            await asyncio.sleep(0.5)

        articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        articles = articles[:self.max_articles]
        logger.info(f"Selected {len(articles)} most relevant articles")

        if articles:
            texts = [a['text'] for a in articles]
            sentiments = self.analyzer.batch_analyze(texts)
            for article, sentiment in zip(articles, sentiments):
                article['sentiment_analysis'] = sentiment
                sentiment_label = sentiment['sentiment']
                opinion = 1 if sentiment_label == 'positive' else -1 if sentiment_label == 'negative' else 0
                if self.publish_to_rabbitmq:
                    try:
                        published_at = (article['publish_date'] if article['publish_date']
                                        else datetime.utcnow())
                        result = self.publisher.publish_news(
                            title=article['title'],
                            symbol=self.ticker,
                            content=article['text'],
                            published_at=published_at,
                            opinion=opinion
                        )
                        if result:
                            logger.info(f"Published to RabbitMQ: {article['title']}")
                        else:
                            logger.error(f"Failed to publish: {article['title']}")
                    except Exception as e:
                        logger.error(f"Error publishing to RabbitMQ: {e}")
        return articles

    async def scrape_to_json(self):
        """Scrape news and return results as JSON."""
        articles = await self.get_news()
        result = {
            "ticker": self.ticker,
            "company": self.company_name,
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "total_articles": len(articles),
            "articles": [
                {
                    "title": a["title"],
                    "source": a["source"],
                    "url": a["url"],
                    "text": a["text"],
                    "sentiment_analysis": a["sentiment_analysis"],
                    "published_at": (a["publish_date"].isoformat() if a["publish_date"]
                                    else datetime.utcnow().isoformat())
                } for a in articles
            ]
        }
        return result

@ns.route('/analyze')
class NewsAnalyzer(Resource):
    @ns.expect(stock_model)
    @ns.marshal_with(response_model, code=200, description='News analysis with sentiment')
    def post(self):
        """Analyze news for a stock ticker."""
        data = request.json
        ticker = data.get('ticker', 'AAPL').upper()
        max_articles = data.get('articles', 5)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            scraper = StockNewsScraper(ticker, max_articles, publish_to_rabbitmq=True)
            result = loop.run_until_complete(scraper.scrape_to_json())
            return result
        except Exception as e:
            logger.exception(f"Error analyzing news for {ticker}: {e}")
            ns.abort(500, f"Error analyzing news: {str(e)}")
        finally:
            loop.close()

@ns.route('/health')
class HealthCheck(Resource):
    def get(self):
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

# Replace @app.before_first_request with a better approach
# Initialize models at startup
def initialize_models():
    """Initialize models on startup."""
    logger.info("Initializing sentiment analyzer...")
    FinBERTSentimentAnalyzer()
    logger.info("Sentiment analyzer initialized.")

if __name__ == "__main__":
    # Call initialize_models directly before running the app
    initialize_models()
    port = int(os.environ.get('PORT', 8001))
    host = os.environ.get('HOST', '0.0.0.0')
    logger.info(f"Starting News Analyzer API on {host}:{port}")
    app.run(host=host, port=port, debug=False)