"""
News analysis and sentiment service.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import os
from textblob import TextBlob
from huggingface_hub import snapshot_download
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import yfinance as yf
import torch

from core import BaseService
from core.utils import get_date_range
from core.logging import logger
from core.progress import create_spinner, print_status, print_error
from core.prometheus_metrics import (
    external_requests_total,
    sentiment_analysis_time_seconds,
)

logger = logger["news"]


class NewsService(BaseService):
    """Service for news analysis and sentiment."""

    _instance = None
    _initialized = False
    _sentiment_analyzer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewsService, cls).__new__(cls)
        return cls._instance

    def ensure_textblob_corpora(self):
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('corpora/wordnet')
            return True
        except LookupError:
            return False

    def __init__(self):
        if self._initialized:
            return

        self.sentiment_analyzer = None
        self.model_version = "0.1.0"
        self.news_data = {}
        self.sentiment_cache = {}
        self.logger = logger

        logger.info("Initializing NewsService...")

        # Download required NLTK data for TextBlob
        try:
            import nltk
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            nltk.download("wordnet")
            logger.info("NLTK data downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {str(e)}")

        # Download TextBlob corpora
        if not self.ensure_textblob_corpora():
            try:
                import subprocess
                subprocess.run(["python", "-m", "textblob.download_corpora"], check=True)
                logger.info("TextBlob corpora downloaded successfully")
            except Exception as e:
                logger.warning(f"Failed to download TextBlob corpora: {str(e)}")
        else:
            logger.info("TextBlob corpora already present")

        # Initialize FinBERT model
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("CUDA available, using GPU")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("MPS available, using Apple Silicon GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("No GPU available, using CPU")

            logger.info("Loading FinBERT model...")
            self.model = BertForSequenceClassification.from_pretrained(
                "yiyanghkust/finbert-tone", num_labels=3
            )
            self.tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.model.to(self.device)

            logger.info("FinBERT model loaded successfully")
            self._sentiment_analyzer = self._analyze_with_finbert
        except Exception as e:
            logger.warning(
                f"Failed to initialize FinBERT, falling back to TextBlob: {str(e)}"
            )
            self._sentiment_analyzer = self._analyze_with_textblob

        self._initialized = True
        logger.info("NewsService initialized successfully")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _download_model(self) -> str:
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_path = os.path.join(cache_dir, "yiyanghkust/finbert-tone")
            if os.path.exists(model_path):
                logger.info(f"Using cached model from {model_path}")
                return model_path
            logger.info(f"Downloading model yiyanghkust/finbert-tone...")
            return snapshot_download(
                "yiyanghkust/finbert-tone",
                force_download=False,
                local_files_only=False,
                cache_dir=cache_dir,
            )
        except Exception as e:
            logger.error(f"Model download attempt failed: {str(e)}")
            raise

    async def initialize(self) -> None:
        try:
            spinner = create_spinner("Initializing sentiment analyzer...")
            spinner.start()
            try:
                model_path = await self._download_model()
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_path,
                    device=0 if torch.cuda.is_available() else -1,
                )
                spinner.stop()
                time.sleep(0.5)
                print_status(
                    "Success",
                    "News service initialized successfully",
                    "success",
                    clear_previous=True,
                )
                self.logger.info("News service initialized successfully")
            except Exception as e:
                spinner.stop()
                self.logger.warning(
                    f"Failed to download model, proceeding with TextBlob fallback: {str(e)}"
                )
                print_status(
                    "Warning",
                    "Using TextBlob fallback for sentiment analysis",
                    "warning",
                    clear_previous=True,
                )
                def textblob_analyzer(text):
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    if polarity > 0.03:
                        label = "POSITIVE"
                    elif polarity < -0.03:
                        label = "NEGATIVE"
                    else:
                        label = "NEUTRAL"
                    confidence = (abs(polarity) + (1 - subjectivity)) / 2
                    return [{"label": label, "score": confidence}]
                self.sentiment_analyzer = textblob_analyzer
        except Exception as e:
            self.logger.error(f"Failed to initialize news service: {str(e)}")
            print_error(e)
            raise

    async def cleanup(self) -> None:
        try:
            self.sentiment_analyzer = None
            self._initialized = False
            self.logger.info("News service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during news service cleanup: {str(e)}")
            print_error(e)

    async def get_news_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting news data retrieval for {symbol}")
            articles = await self._get_news_articles(symbol, start_date, end_date)
            self.logger.info(f"Retrieved {len(articles)} articles for {symbol}")
            if not articles:
                self.logger.info(f"No articles found for {symbol}")
                return {
                    "articles": [],
                    "total_articles": 0,
                    "sentiment_metrics": {},
                    "meta": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "version": self.model_version,
                        "message": "No news articles found",
                        "documentation": "/docs",
                        "endpoints": ["/news/"],
                    },
                }
            start_time = time.perf_counter()
            processed_articles = []
            for article in articles:
                try:
                    sentiment = None
                    confidence = None
                    if self.sentiment_analyzer is not None:
                        try:
                            text = article["title"] + " " + article["content"]
                            sentiment_result = self.sentiment_analyzer(text)[0]
                            sentiment = sentiment_result["label"]
                            confidence = sentiment_result["score"]
                        except Exception as e:
                            self.logger.warning(f"Sentiment analysis failed: {str(e)}")
                    processed_article = {
                        "title": article["title"],
                        "url": article["url"],
                        "published_date": article["published_date"].isoformat() if article["published_date"] else None,
                        "source": article["source"],
                        "sentiment": sentiment,
                        "confidence": confidence,
                    }
                    processed_articles.append(processed_article)
                except Exception as e:
                    self.logger.error(f"Error processing article: {str(e)}")
                    continue
            sentiment_analysis_duration = time.perf_counter() - start_time
            sentiment_analysis_time_seconds.labels(
                number_articles=len(processed_articles)
            ).observe(sentiment_analysis_duration)
            self.logger.info(
                f"Successfully processed {len(processed_articles)} articles"
            )
            metrics = self._calculate_metrics(processed_articles)
            self.logger.debug(f"Calculated metrics: {metrics}")
            result = {
                "articles": processed_articles,
                "total_articles": len(processed_articles),
                "sentiment_metrics": metrics,
                "meta": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "version": self.model_version,
                    "message": "News articles retrieved successfully",
                    "documentation": "/docs",
                    "endpoints": ["/news/"],
                },
            }
            self.logger.info(f"Successfully completed news data retrieval for {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting news data: {str(e)}")
            raise

    async def _get_news_articles(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        try:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            end_date = end_date.replace(
                hour=23, minute=59, second=59, microsecond=999999
            )
            ticker = yf.Ticker(symbol)
            news = ticker.news
            external_requests_total.labels(site="yahoo_finance", result="success").inc()
            articles = []
            for item in news:
                try:
                    news_date = None
                    if "pubDate" in item:
                        news_date = datetime.fromisoformat(
                            item["pubDate"].replace("Z", "+00:00")
                        )
                    elif "content" in item and "pubDate" in item["content"]:
                        news_date = datetime.fromisoformat(
                            item["content"]["pubDate"].replace("Z", "+00:00")
                        )
                    if news_date and start_date <= news_date <= end_date:
                        article = {}
                        if "content" in item:
                            content = item["content"]
                            article["title"] = content.get("title", "No Title")
                            article["content"] = content.get("summary", "")
                            article["published_date"] = news_date
                            if (
                                "clickThroughUrl" in content
                                and content["clickThroughUrl"]
                                and "url" in content["clickThroughUrl"]
                            ):
                                article["url"] = content["clickThroughUrl"]["url"]
                            elif (
                                "canonicalUrl" in content
                                and content["canonicalUrl"]
                                and "url" in content["canonicalUrl"]
                            ):
                                article["url"] = content["canonicalUrl"]["url"]
                            else:
                                article["url"] = ""
                            if "provider" in content and content["provider"]:
                                article["source"] = content["provider"].get(
                                    "displayName", "Unknown"
                                )
                            else:
                                article["source"] = "Unknown"
                        else:
                            article["title"] = item.get("title", "No Title")
                            article["content"] = item.get("summary", "")
                            article["published_date"] = news_date
                            article["url"] = item.get("link", "")
                            article["source"] = "Unknown"
                        articles.append(article)
                except Exception as e:
                    self.logger.error(f"Error processing news item: {str(e)}")
                    external_requests_total.labels(
                        site="yahoo_finance", result="error"
                    ).inc()
                    continue
            return articles
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []

    def _calculate_metrics(
        self, sentiment_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not sentiment_results:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "average_confidence": 0.0,
            }
        total = len(sentiment_results)
        positive = sum(
            result.get("confidence")
            for result in sentiment_results
            if result.get("sentiment") == "POSITIVE"
            and result.get("confidence") is not None
        )
        negative = sum(
            result.get("confidence")
            for result in sentiment_results
            if result.get("sentiment") == "NEGATIVE"
            and result.get("confidence") is not None
        )
        neutral = sum(
            result.get("confidence")
            for result in sentiment_results
            if result.get("sentiment") == "NEUTRAL"
            and result.get("confidence") is not None
        )
        valid_confidences = [
            r["confidence"]
            for r in sentiment_results
            if r.get("confidence") is not None
        ]
        avg_confidence = (
            sum(valid_confidences) / len(valid_confidences)
            if valid_confidences
            else 0.0
        )
        return {
            "positive": positive / total if total > 0 else 0.0,
            "negative": negative / total if total > 0 else 0.0,
            "neutral": neutral / total if total > 0 else 0.0,
            "average_confidence": avg_confidence,
        }

    def _analyze_with_finbert(self, text: str) -> List[Dict]:
        try:
            max_length = 512
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            sentiment_map = {0: "neutral", 1: "positive", 2: "negative"}
            sentiment_idx = torch.argmax(probs).item()
            sentiment = sentiment_map[sentiment_idx]
            confidence = probs[sentiment_idx].item()
            return [
                {
                    "label": sentiment.upper(),
                    "score": confidence,
                }
            ]
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return self._analyze_with_textblob(text)

    def _analyze_with_textblob(self, text: str) -> List[Dict]:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            confidence = (abs(polarity) + (1 - subjectivity)) / 2
            return [
                {
                    "label": sentiment.upper(),
                    "score": confidence,
                }
            ]
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {e}")
            return [
                {
                    "label": "NEUTRAL",
                    "score": 1.0,
                }
            ]
