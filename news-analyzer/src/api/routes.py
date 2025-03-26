from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from src.services.news_service import NewsService
from src.services.sentiment_service import SentimentService
from src.core.config import Settings
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel, Field
from ..core.logging import setup_logging

router = APIRouter()
settings = Settings()
news_service = NewsService(settings)
sentiment_service = SentimentService()

@router.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

class ArticleResponse(BaseModel):
    url: str
    title: str
    ticker: str
    content: str
    date: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]
    opinion: int
    summary: str

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.post("/analyze", response_model=List[ArticleResponse])
async def analyze_news(
    ticker: str = Query(
        ...,
        description="Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)",
        example="AAPL",
        min_length=1,
        max_length=5
    )
):
    """Analyze news for a given stock ticker"""
    try:
        # Fetch news articles
        articles = await news_service.get_news(ticker)
        
        # Extract text content for sentiment analysis
        texts = [article["content"] for article in articles]
        
        # Analyze sentiment
        sentiment_results = await sentiment_service.analyze_sentiment(texts)
        
        # Combine articles with sentiment analysis
        results = []
        for article, sentiment in zip(articles, sentiment_results):
            result = {
                "url": article["url"],
                "title": article["title"],
                "ticker": article["ticker"],
                "content": article["content"],
                "date": article["date"],
                "sentiment": sentiment["sentiment"],
                "confidence": sentiment["confidence"],
                "scores": sentiment["scores"],
                "opinion": sentiment["opinion"],
                "summary": sentiment["summary"]
            }
            results.append(result)
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/{symbol}")
async def get_sentiment(
    symbol: str,
    news_service: NewsService = Depends(lambda: NewsService(Settings())),
    sentiment_service: SentimentService = Depends(lambda: SentimentService())
):
    """Get sentiment analysis for a stock symbol."""
    try:
        articles = await news_service.get_news(symbol)
        if not articles:
            return {"symbol": symbol, "sentiment_analysis": []}

        sentiments = await sentiment_service.analyze_sentiment([article["content"] for article in articles])
        return {"symbol": symbol, "sentiment_analysis": sentiments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 