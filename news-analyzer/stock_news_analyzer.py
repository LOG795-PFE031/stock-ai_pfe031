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
from np3k_group_crawler import crawl_and_extract
from deepcrawler import get_todays_news_urls
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

# Define output model for articles
article_model = ns.model('ArticleModel', {
    'url': fields.String(description='Article URL'),
    'title': fields.String(description='Article title'),
    'ticker': fields.String(description='Stock ticker'),
    'content': fields.String(description='Article content'),
    'date': fields.String(description='Publication date'),
    'opinion': fields.Integer(description='Sentiment score (-1: negative, 0: neutral, 1: positive)')
})

# FinBERT Sentiment Analyzer class
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
            # BERT models have a maximum token limit of 512 tokens
            max_length = 512
            
            # Tokenize and truncate texts to the maximum sequence length
            encoded_texts = []
            for text in texts:
                # Tokenize with truncation
                encoded = self.tokenizer.encode_plus(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                encoded_texts.append(encoded)
            
            # Process with the model
            results = []
            for i, encoded in enumerate(encoded_texts):
                # Send to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get probabilities and sentiment
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
                # Map indices to sentiments
                sentiment_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
                sentiment_idx = torch.argmax(probs).item()
                sentiment = sentiment_map[sentiment_idx]
                confidence = probs[sentiment_idx].item()
                
                results.append({
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "scores": {
                        "positive": probs[1].item(),
                        "negative": probs[2].item(),
                        "neutral": probs[0].item()
                    }
                })
            
            return results
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Return neutral sentiment for all texts in case of failure
            return [{"sentiment": "neutral", "confidence": 1.0, "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}} for _ in texts]

# Stock News Scraper class
class StockNewsScraper:
    def __init__(self, ticker, urls, publish_to_rabbitmq=False):
        self.ticker = ticker
        self.urls = urls
        self.publish_to_rabbitmq = publish_to_rabbitmq
        self.company_name = "Unknown Company"  # Placeholder; could fetch from yfinance if needed

    async def get_news(self):
        """Fetch and analyze news articles using deepcrawler and np3k_group_crawler."""
        # Create a list of tasks for each URL with url and ticker arguments
        tasks = [crawl_and_extract(url, self.ticker) for url in self.urls]
        # Run all tasks concurrently
        articles_json = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out any failed crawls or exceptions
        articles_json = [article for article in articles_json if isinstance(article, dict) and article]
        if not articles_json:
            logger.warning("No articles extracted.")
            return []

        # Extract text content for sentiment analysis
        texts = []
        for article in articles_json:
            # Make sure text isn't too long for BERT model (which has 512 token limit)
            # Truncate to around 2000 chars which should be well under the token limit
            text = article.get('content', '')
            if len(text) > 2000:
                text = text[:2000]
            texts.append(text)
            
        analyzer = FinBERTSentimentAnalyzer()
        sentiments = analyzer.batch_analyze(texts)

        # Combine articles with sentiment results
        for article, sentiment in zip(articles_json, sentiments):
            article['sentiment_analysis'] = sentiment
        return articles_json

    async def scrape_to_json(self):
        """Scrape news and return a list of articles in the desired format."""
        articles = await self.get_news()
        articles_list = [
            {
                "url": a.get("url", ""),
                "title": a.get("title", "Untitled"),
                "ticker": self.ticker,
                "content": a.get("content", ""),
                "date": a.get("date", "Unknown"),
                "opinion": self.label_to_score(a.get("sentiment_analysis", {}).get("sentiment", "neutral"))
            } for a in articles
        ]
        return articles_list

    def label_to_score(self, label):
        """Map sentiment label to a score (-1, 0, 1)."""
        mapping = {"positive": 1, "neutral": 0, "negative": -1}
        return mapping.get(label.lower(), 0)  # Default to 0 if label is unknown

@ns.route('/analyze')
class NewsAnalyzer(Resource):
    @ns.expect(api.model('InputModel', {
        'ticker': fields.String(required=True, example='AAPL')
    }), validate=True)  # Add input validation
    @ns.marshal_list_with(article_model)
    def post(self):
        """Analyze news for a stock ticker."""
        try:
            data = request.get_json(force=True)
            ticker = data.get('ticker', 'AAPL').upper()
            
            if not ticker:
                ns.abort(400, "Missing required 'ticker' field")

            urls, ticker = get_todays_news_urls(ticker)
        

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
            try:
                scraper = StockNewsScraper(ticker, urls, publish_to_rabbitmq=True)
                result = loop.run_until_complete(scraper.scrape_to_json())
                return result
            except Exception as e:
                logger.exception(f"Error analyzing news for {ticker}: {e}")
                ns.abort(500, f"Error analyzing news: {str(e)}")
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Request data: {request.data}")  # Log raw request data
            logger.exception(f"Bad request error: {str(e)}")
            ns.abort(400, f"Invalid request: {str(e)}")


@ns.route('/health')
class HealthCheck(Resource):
    def get(self):
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

# Initialize models at startup
def initialize_models():
    """Initialize models on startup."""
    logger.info("Initializing sentiment analyzer...")
    FinBERTSentimentAnalyzer()
    logger.info("Sentiment analyzer initialized.")

if __name__ == "__main__":
    initialize_models()
    port = int(os.environ.get('PORT', 8092))
    host = os.environ.get('HOST', '0.0.0.0')
    logger.info(f"Starting News Analyzer API on {host}:{port}")
    app.run(host=host, port=port, debug=False)