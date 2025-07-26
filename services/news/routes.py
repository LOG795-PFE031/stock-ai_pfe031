from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from datetime import datetime
from .news_service import NewsService
from services.news.schemas import NewsDataResponse, MetaInfo, HealthResponse
from core.config import config
from core.utils import get_date_range
from core.logging import logger

router = APIRouter()
api_logger = logger["news"]
news_service = NewsService()

@router.get("/", response_class=RedirectResponse, tags=["System"])
async def root():
    return "/docs"

#API welcome message
@router.get("/welcome", response_model=MetaInfo, tags=["System"])
async def api_welcome():
    return {
        "message": "Welcome to Stock AI News Service API",
        "version": config.api.API_VERSION,
        "documentation": "/docs",
        "endpoints": ["/news/"],
    }

# Health check endpoint
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of all services."""
    try:
        # Import services from main to avoid circular imports
        from .main import news_service

        # Check each service's health
        news_health = await news_service.health_check()

        # Create response with boolean values
        return HealthResponse(
            status=(
                "healthy" if news_health["status"] == "healthy" else "unhealthy"
            ),
            components={"news_service": news_health["status"] == "healthy"},
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Health check failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(exception)}"
        ) from exception
    

@router.get("/news/", response_model=NewsDataResponse, tags=["News"])
async def get_news_data(
    symbol: str = Query(..., description="Stock symbol to retrieve news data for"),
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)")
):
    try:
        # Validate symbol (simple check)
        if not symbol.isalnum():
            raise HTTPException(status_code=400, detail=f"Invalid stock symbol: {symbol}")
        start, end = get_date_range(start_date, end_date)
        data = await news_service.get_news_data(symbol, start, end)
        return NewsDataResponse(
            symbol=symbol,
            articles=data["articles"],
            total_articles=data["total_articles"],
            sentiment_metrics=data["sentiment_metrics"],
            meta=MetaInfo(
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                version=data["meta"].get("version", "1.0.0"),
                message=data["meta"].get("message", "News data retrieved successfully"),
                documentation=data["meta"].get("documentation", "/docs"),
                endpoints=data["meta"].get("endpoints", ["/news/"]),
            ),
        )
    except Exception as e:
        api_logger.error(f"Failed to get news data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get news data: {e}") 
