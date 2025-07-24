from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from datetime import datetime
import logging
from news_service import NewsService
from schemas import NewsDataResponse, MetaInfo
from utils import get_date_range

router = APIRouter()
logger = logging.getLogger("news_service")
news_service = NewsService()

@router.get("/", response_class=RedirectResponse, tags=["System"])
async def root():
    return "/docs"

@router.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy"}

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
        logger.error(f"Failed to get news data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get news data: {e}") 
