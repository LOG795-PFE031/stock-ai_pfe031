from typing import List, Dict, Optional
from pydantic import BaseModel

class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    components: Dict[str, bool]

class MetaInfo(BaseModel):
    """Metadata about the news service."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    version: Optional[str] = None
    message: Optional[str] = None
    documentation: Optional[str] = None
    endpoints: Optional[List[str]] = None

class NewsArticle(BaseModel):
    """Schema for a news article."""
    title: str
    url: Optional[str] = None
    published_date: Optional[str] = None
    source: Optional[str] = None
    sentiment: Optional[str] = None
    confidence: Optional[float] = None

class NewsDataResponse(BaseModel):
    """Response schema for news data."""
    symbol: str
    articles: List[NewsArticle]
    total_articles: int
    sentiment_metrics: Dict[str, float]
    meta: MetaInfo 
