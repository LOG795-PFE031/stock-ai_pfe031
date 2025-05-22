"""
News analysis and sentiment service.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import os
from textblob import TextBlob
import numpy as np
from huggingface_hub import snapshot_download
from rich.panel import Panel
import time
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import yfinance as yf
import logging
import torch

from services.base_service import BaseService
from core.utils import get_date_range
from core.logging import logger
from core.progress import (
    create_spinner,
    print_status,
    print_error,
    create_layout,
    update_layout,
)

logger = logging.getLogger(__name__)


class NewsService(BaseService):
    """Service for news analysis and sentiment."""

    _instance = None
    _initialized = False
    _sentiment_analyzer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewsService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        super().__init__()
        self.sentiment_analyzer = None
        self.model_version = "0.1.0"
        self.news_data = {}
        self.sentiment_cache = {}
        self.logger = logging.getLogger(__name__)
        self.layout = create_layout()

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
        try:
            import subprocess

            subprocess.run(["python", "-m", "textblob.download_corpora"], check=True)
            logger.info("TextBlob corpora downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download TextBlob corpora: {str(e)}")

        # Initialize FinBERT model
        try:
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
            self.model = BertForSequenceClassification.from_pretrained(
                "yiyanghkust/finbert-tone", num_labels=3
            )
            self.tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.model.to(self.device)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
            )
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
        """Download the sentiment model with retry logic."""
        try:
            # First try to load from cache
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_path = os.path.join(cache_dir, self.config.model.SENTIMENT_MODEL_NAME)

            if os.path.exists(model_path):
                logger.info(f"Using cached model from {model_path}")
                return model_path

            # If not in cache, download
            logger.info(
                f"Downloading model {self.config.model.SENTIMENT_MODEL_NAME}..."
            )
            return snapshot_download(
                self.config.model.SENTIMENT_MODEL_NAME,
                force_download=False,  # Don't force download if already cached
                local_files_only=False,
                cache_dir=cache_dir,
            )
        except Exception as e:
            logger.error(f"Model download attempt failed: {str(e)}")
            raise

    async def initialize(self) -> None:
        """Initialize the news service."""
        try:
            # Create spinner
            spinner = create_spinner("Initializing sentiment analyzer...")

            # Start spinner
            spinner.start()

            try:
                # Try to download model with retry logic
                model_path = await self._download_model()

                # Initialize sentiment analyzer
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_path,
                    device=0 if self.config.model.USE_GPU else -1,
                )

                # Stop spinner
                spinner.stop()

                # Clear console and show success message
                time.sleep(0.5)  # Small delay to ensure spinner is cleared
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

                # Initialize TextBlob as fallback with improved confidence calculation
                def textblob_analyzer(text):
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity

                    # Enhanced financial context words and phrases
                    positive_words = [
                        "buy",
                        "bull",
                        "soar",
                        "growth",
                        "opportunity",
                        "magnificent",
                        "brilliant",
                        "steady",
                        "top",
                        "best",
                        "strong",
                        "outperform",
                        "upgrade",
                        "recommend",
                        "favorite",
                        "leading",
                        "dominant",
                        "innovative",
                        "breakthrough",
                        "revolutionary",
                        "transformative",
                        "promising",
                        "undervalued",
                        "bargain",
                        "attractive",
                        "compelling",
                        "conviction",
                        "long-term",
                        "sustainable",
                    ]

                    negative_words = [
                        "sell",
                        "bear",
                        "dip",
                        "turmoil",
                        "risk",
                        "beaten-down",
                        "down",
                        "collapse",
                        "crash",
                        "warning",
                        "concern",
                        "caution",
                        "trouble",
                        "struggle",
                        "challenge",
                        "headwind",
                        "pressure",
                        "decline",
                        "drop",
                        "fall",
                        "plunge",
                        "slump",
                        "weakness",
                        "vulnerable",
                        "exposed",
                        "threat",
                        "overvalued",
                        "expensive",
                        "premium",
                        "bubble",
                        "speculative",
                        "uncertain",
                        "volatile",
                    ]

                    # Count occurrences of financial sentiment words
                    text_lower = text.lower()
                    positive_count = sum(
                        1 for word in positive_words if word.lower() in text_lower
                    )
                    negative_count = sum(
                        1 for word in negative_words if word.lower() in text_lower
                    )

                    # Analyze phrases for stronger sentiment signals
                    phrases = blob.noun_phrases
                    phrase_sentiment = 0
                    for phrase in phrases:
                        phrase_lower = phrase.lower()
                        if any(word in phrase_lower for word in positive_words):
                            phrase_sentiment += 0.2
                        if any(word in phrase_lower for word in negative_words):
                            phrase_sentiment -= 0.2

                    # Calculate context adjustment
                    word_adjustment = (positive_count - negative_count) * 0.1
                    context_adjustment = word_adjustment + phrase_sentiment

                    # Adjust polarity based on context
                    adjusted_polarity = polarity + context_adjustment

                    # Determine sentiment label with adjusted thresholds
                    if adjusted_polarity > 0.03:  # Even lower threshold for positive
                        label = "POSITIVE"
                    elif adjusted_polarity < -0.03:  # Even lower threshold for negative
                        label = "NEGATIVE"
                    else:
                        label = "NEUTRAL"

                    # Calculate confidence score
                    # Base confidence on adjusted polarity and subjectivity
                    base_confidence = abs(adjusted_polarity)

                    # Increase confidence if there are clear financial sentiment words
                    word_confidence = min(1.0, (positive_count + negative_count) * 0.1)

                    # Add phrase confidence
                    phrase_confidence = min(1.0, abs(phrase_sentiment))

                    # Combine all confidence factors
                    confidence = (
                        base_confidence
                        + word_confidence
                        + phrase_confidence
                        + (1 - subjectivity)
                    ) / 4

                    # Ensure confidence is between 0 and 1
                    confidence = max(0.0, min(1.0, confidence))

                    # Log analysis details for debugging
                    self.logger.debug(
                        f"""
                    Text: {text[:100]}...
                    Polarity: {polarity}
                    Subjectivity: {subjectivity}
                    Positive words: {positive_count}
                    Negative words: {negative_count}
                    Phrase sentiment: {phrase_sentiment}
                    Context adjustment: {context_adjustment}
                    Adjusted polarity: {adjusted_polarity}
                    Label: {label}
                    Confidence: {confidence}
                    """
                    )

                    return [{"label": label, "score": confidence}]

                self.sentiment_analyzer = textblob_analyzer

        except Exception as e:
            self.logger.error(f"Failed to initialize news service: {str(e)}")
            print_error(e)
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear sentiment analyzer
            self.sentiment_analyzer = None
            self._initialized = False
            self.logger.info("News service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during news service cleanup: {str(e)}")
            print_error(e)

    async def analyze_news(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyze news articles for a given symbol.

        Args:
            symbol: Stock symbol
            days: Number of days of news to analyze

        Returns:
            Dictionary containing news analysis results
        """
        if not self._initialized:
            raise RuntimeError("News service not initialized")

        try:
            with create_spinner(f"Analyzing news for {symbol}...") as spinner:
                # Get date range
                start_date, end_date = get_date_range(days)

                # Get news articles
                articles = await self._get_news_articles(symbol, start_date, end_date)

                # Analyze sentiment
                sentiment_results = await self._analyze_sentiment(articles)

                # Calculate aggregate metrics
                metrics = self._calculate_metrics(sentiment_results)

                return {
                    "symbol": symbol,
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                    },
                    "total_articles": len(articles),
                    "sentiment_metrics": metrics,
                    "articles": sentiment_results,
                    "model_version": self.model_version,
                }

        except Exception as e:
            self.logger.error(f"Error analyzing news for {symbol}: {str(e)}")
            print_error(e)
            raise

    async def _get_news_articles(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get news articles for a given symbol and date range."""
        try:
            # Ensure timezone-aware datetimes
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            # Fetch news from Yahoo Finance
            ticker = yf.Ticker(symbol)
            news = ticker.news

            articles = []
            for item in news:
                try:
                    # Extract article date
                    news_date = None
                    if "displayTime" in item:
                        news_date = datetime.fromisoformat(
                            item["displayTime"].replace("Z", "+00:00")
                        )
                    elif "content" in item and "displayTime" in item["content"]:
                        news_date = datetime.fromisoformat(
                            item["content"]["displayTime"].replace("Z", "+00:00")
                        )

                    # Only include articles within the date range
                    if news_date and start_date <= news_date <= end_date:
                        article = {}
                        if "content" in item:
                            content = item["content"]
                            article["title"] = content.get("title", "No Title")
                            article["content"] = content.get("summary", "")
                            article["published_date"] = news_date

                            # Get URL
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

                            # Get source
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
                    continue

            return articles

        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []

    async def _analyze_sentiment(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze sentiment for a list of articles."""
        results = []
        for article in articles:
            try:
                # Truncate text if needed
                text = article["title"] + " " + article["content"]
                if len(text) > self.config.model.MAX_NEWS_LENGTH:
                    text = text[: self.config.model.MAX_NEWS_LENGTH]

                # Get sentiment analysis
                sentiment = self.sentiment_analyzer(text)[0]

                results.append(
                    {
                        "title": article["title"],
                        "url": article["url"],
                        "published_date": article["published_date"],
                        "sentiment": sentiment["label"],
                        "confidence": sentiment["score"],
                    }
                )
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment for article: {str(e)}")
                continue

        return results

    def _calculate_metrics(
        self, sentiment_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate aggregate sentiment metrics."""
        if not sentiment_results:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "average_confidence": 0.0,
            }

        total = len(sentiment_results)

        # positive = sum(1 for r in sentiment_results if r.get("sentiment") == "POSITIVE")
        positive = sum(
            result.get("confidence")
            for result in sentiment_results
            if result.get("sentiment") == "POSITIVE"
            and result.get("confidence") is not None
        )

        # negative = sum(1 for r in sentiment_results if r.get("sentiment") == "NEGATIVE")
        negative = sum(
            result.get("confidence")
            for result in sentiment_results
            if result.get("sentiment") == "NEGATIVE"
            and result.get("confidence") is not None
        )

        # neutral = total - positive - negative
        neutral = sum(
            result.get("confidence")
            for result in sentiment_results
            if result.get("sentiment") == "NEUTRAL"
            and result.get("confidence") is not None
        )

        # Calculate average confidence only for articles with valid confidence scores
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

    async def get_news_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Get news data for a given symbol and date range."""
        try:
            self.logger.info(f"Starting news data retrieval for {symbol}")

            # Fetch news articles
            articles = await self._get_news_articles(symbol, start_date, end_date)
            self.logger.info(f"Retrieved {len(articles)} articles for {symbol}")

            if not articles:
                self.logger.info(f"No articles found for {symbol}")
                return {
                    "articles": [],
                    "total_articles": 0,
                    "meta": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "version": self.model_version,
                        "message": "No news articles found",
                        "documentation": "https://api.example.com/docs",
                        "endpoints": ["/api/data/news/{symbol}"],
                    },
                }

            # Process and analyze articles
            processed_articles = []
            for article in articles:
                try:
                    self.logger.debug(
                        f"Processing article: {article.get('title', 'No title')}"
                    )

                    # Get sentiment analysis if available
                    sentiment = None
                    confidence = None
                    if self.sentiment_analyzer is not None:
                        try:
                            text = article["title"] + " " + article["content"]
                            self.logger.debug(f"Analyzing text: {text[:100]}...")

                            sentiment_result = self.sentiment_analyzer(text)[0]
                            sentiment = sentiment_result["label"]
                            confidence = sentiment_result["score"]

                            self.logger.debug(
                                f"Sentiment analysis result: {sentiment} (confidence: {confidence})"
                            )
                        except Exception as e:
                            self.logger.warning(f"Sentiment analysis failed: {str(e)}")

                    processed_article = {
                        "title": article["title"],
                        "url": article["url"],
                        "published_date": article["published_date"].isoformat(),
                        "source": article["source"],
                        "sentiment": sentiment,
                        "confidence": confidence,
                    }
                    processed_articles.append(processed_article)
                except Exception as e:
                    self.logger.error(f"Error processing article: {str(e)}")
                    continue

            self.logger.info(
                f"Successfully processed {len(processed_articles)} articles"
            )

            # Calculate sentiment metrics
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
                    "documentation": "https://api.example.com/docs",
                    "endpoints": ["/api/data/news/{symbol}"],
                },
            }

            self.logger.info(f"Successfully completed news data retrieval for {symbol}")
            return result

        except Exception as e:
            self.logger.error(f"Error getting news data: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error(f"Error traceback: {e.__traceback__}")
            raise

    def _analyze_with_finbert(self, text: str) -> List[Dict]:
        """Analyze sentiment using FinBERT model"""
        try:
            # BERT models have a maximum token limit of 512 tokens
            max_length = 512

            # Tokenize with truncation
            encoded = self.tokenizer.encode_plus(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Send to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Get probabilities and sentiment
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            # Map indices to sentiments
            sentiment_map = {0: "neutral", 1: "positive", 2: "negative"}
            sentiment_idx = torch.argmax(probs).item()
            sentiment = sentiment_map[sentiment_idx]
            confidence = probs[sentiment_idx].item()

            # Convert to opinion score
            opinion_map = {"positive": 1, "negative": -1, "neutral": 0}
            opinion = opinion_map.get(sentiment, 0)

            # Generate summary
            confidence_percentage = round(confidence * 100)
            if confidence_percentage >= 80:
                strength = "strongly"
            elif confidence_percentage >= 60:
                strength = "moderately"
            else:
                strength = "slightly"
            summary = f"{strength.capitalize()} {sentiment} sentiment ({confidence_percentage}% confidence)"

            return [
                {
                    "label": sentiment.upper(),
                    "score": confidence,
                    "scores": {
                        "positive": probs[1].item(),
                        "negative": probs[2].item(),
                        "neutral": probs[0].item(),
                    },
                    "opinion": opinion,
                    "summary": summary,
                }
            ]
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return self._analyze_with_textblob(text)

    def _analyze_with_textblob(self, text: str) -> List[Dict]:
        """Fallback sentiment analysis using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Determine sentiment label
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Calculate confidence
            confidence = (abs(polarity) + (1 - subjectivity)) / 2

            # Convert to opinion score
            opinion_map = {"positive": 1, "negative": -1, "neutral": 0}
            opinion = opinion_map.get(sentiment, 0)

            # Generate summary
            confidence_percentage = round(confidence * 100)
            if confidence_percentage >= 80:
                strength = "strongly"
            elif confidence_percentage >= 60:
                strength = "moderately"
            else:
                strength = "slightly"
            summary = f"{strength.capitalize()} {sentiment} sentiment ({confidence_percentage}% confidence)"

            return [
                {
                    "label": sentiment.upper(),
                    "score": confidence,
                    "scores": {
                        "positive": 1.0 if sentiment == "positive" else 0.0,
                        "negative": 1.0 if sentiment == "negative" else 0.0,
                        "neutral": 1.0 if sentiment == "neutral" else 0.0,
                    },
                    "opinion": opinion,
                    "summary": summary,
                }
            ]
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {e}")
            return [
                {
                    "label": "NEUTRAL",
                    "score": 1.0,
                    "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                    "opinion": 0,
                    "summary": "Unable to analyze sentiment",
                }
            ]
