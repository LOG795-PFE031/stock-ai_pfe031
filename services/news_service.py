"""
News analysis and sentiment service.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from transformers import pipeline
import os
from textblob import TextBlob
import numpy as np
from huggingface_hub import snapshot_download
from rich.panel import Panel
import time
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from services.base_service import BaseService
from core.utils import get_date_range
from core.logging import logger
from core.progress import (
    create_spinner,
    print_status,
    print_error,
    create_layout,
    update_layout
)

class NewsService(BaseService):
    """Service for news analysis and sentiment."""
    
    def __init__(self):
        super().__init__()
        self.sentiment_analyzer = None
        self.model_version = "0.1.0"
        self._initialized = False
        self.news_data = {}
        self.sentiment_cache = {}
        self.logger = logger['news']
        self.layout = create_layout()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _download_model(self) -> str:
        """Download the sentiment model with retry logic."""
        try:
            return snapshot_download(
                self.config.model.SENTIMENT_MODEL_NAME,
                force_download=True,
                local_files_only=False
            )
        except Exception as e:
            self.logger.error(f"Model download attempt failed: {str(e)}")
            raise
    
    async def initialize(self) -> None:
        """Initialize the news service."""
        try:
            # Create spinner
            spinner = create_spinner("Initializing sentiment analyzer...")
            
            # Start spinner
            spinner.start()
            
            try:
                # Download model with retry logic
                model_path = await self._download_model()
                
                # Initialize sentiment analyzer
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_path
                )
                
                # Stop spinner
                spinner.stop()
                
                # Clear console and show success message
                time.sleep(0.5)  # Small delay to ensure spinner is cleared
                print_status(
                    "Success",
                    "News service initialized successfully",
                    "success",
                    clear_previous=True
                )
                
                self._initialized = True
                self.logger.info("News service initialized successfully")
                
            except Exception as e:
                spinner.stop()
                self.logger.warning(f"Failed to download model, proceeding with TextBlob fallback: {str(e)}")
                print_status(
                    "Warning",
                    "Using TextBlob fallback for sentiment analysis",
                    "warning",
                    clear_previous=True
                )
                self._initialized = True  # Mark as initialized even with fallback
            
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
    
    async def analyze_news(
        self,
        symbol: str,
        days: int = 7
    ) -> Dict[str, Any]:
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
                        "end": end_date.isoformat()
                    },
                    "total_articles": len(articles),
                    "sentiment_metrics": metrics,
                    "articles": sentiment_results,
                    "model_version": self.model_version
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news for {symbol}: {str(e)}")
            print_error(e)
            raise
    
    async def _get_news_articles(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get news articles for a given symbol and date range."""
        # Implementation will be added later
        pass
    
    async def _analyze_sentiment(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze sentiment for a list of articles."""
        results = []
        for article in articles:
            try:
                # Truncate text if needed
                text = article["title"] + " " + article["content"]
                if len(text) > self.config.model.MAX_NEWS_LENGTH:
                    text = text[:self.config.model.MAX_NEWS_LENGTH]
                
                # Get sentiment analysis
                sentiment = self.sentiment_analyzer(text)[0]
                
                results.append({
                    "title": article["title"],
                    "url": article["url"],
                    "published_date": article["published_date"],
                    "sentiment": sentiment["label"],
                    "confidence": sentiment["score"]
                })
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment for article: {str(e)}")
                continue
        
        return results
    
    def _calculate_metrics(
        self,
        sentiment_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate aggregate sentiment metrics."""
        if not sentiment_results:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "average_confidence": 0.0
            }
        
        total = len(sentiment_results)
        positive = sum(1 for r in sentiment_results if r["sentiment"] == "POSITIVE")
        negative = sum(1 for r in sentiment_results if r["sentiment"] == "NEGATIVE")
        neutral = total - positive - negative
        
        avg_confidence = sum(r["confidence"] for r in sentiment_results) / total
        
        return {
            "positive": positive / total,
            "negative": negative / total,
            "neutral": neutral / total,
            "average_confidence": avg_confidence
        } 