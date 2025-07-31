from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from datetime import datetime, timezone
from typing import List
import pandas as pd

from .data_service import DataService
from .schemas import (
    StockDataResponse,
    CurrentPriceResponse,
    CleanupResponse,
    MetaInfo,
    HealthResponse,
    StockInfo,
    StockPrice,
    StocksListDataResponse,
    StockItem,
)
from core.config import config
from core.logging import logger

router = APIRouter()
api_logger = logger["data"]
data_service = DataService()


@router.get("/", response_class=RedirectResponse, tags=["System"])
async def root():
    return "/docs"


# API welcome message
@router.get("/welcome", response_model=MetaInfo, tags=["System"])
async def api_welcome():
    return {
        "message": "Welcome to Stock AI Data Service API",
        "version": config.api.API_VERSION,
        "documentation": "/docs",
        "endpoints": [
            "/stocks",
            "/stock/current",
            "/stock/historical",
            "/stock/recent",
            "/stock/from-end-date",
            "/cleanup",
        ],
    }


# Health check endpoint
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of all services."""
    try:
        # Import services from main to avoid circular imports
        from .main import data_service

        # Check each service's health
        data_health = await data_service.health_check()

        # Create response with boolean values
        return HealthResponse(
            status=("healthy" if data_health["status"] == "healthy" else "unhealthy"),
            components={"data_service": data_health["status"] == "healthy"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Health check failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(exception)}"
        ) from exception


@router.get("/stocks", response_model=StocksListDataResponse, tags=["Data"])
async def get_stocks():
    """Get list of available stocks from NASDAQ-100, sorted by absolute percentage change (top movers first)."""
    try:
        response = await data_service.get_nasdaq_stocks()

        # Extract the list of stocks from the response
        if "data" in response and isinstance(response["data"], list):
            stock_items = []
            for stock in response["data"]:
                # Create StockItem with available data, providing defaults for missing fields
                stock_item = StockItem(
                    symbol=stock.get("symbol", ""),
                    sector=stock.get("sector", "Unknown"),  # Provide default if missing
                    companyName=stock.get(
                        "name", stock.get("companyName", "")
                    ),  # Use name if companyName not available
                    marketCap=stock.get(
                        "marketCap", "N/A"
                    ),  # Provide default if missing
                    lastSalePrice=stock.get("lastSalePrice", "0.00"),
                    netChange=stock.get(
                        "netChange", "0.00"
                    ),  # Provide default if missing
                    percentageChange=stock.get("percentageChange", "0.00%"),
                    deltaIndicator=stock.get(
                        "deltaIndicator", ""
                    ),  # Provide default if missing
                )
                stock_items.append(stock_item)

            # Return the proper response format
            return StocksListDataResponse(count=len(stock_items), data=stock_items)
        else:
            raise ValueError("Invalid response format from NASDAQ API")
    except Exception as e:
        api_logger.error(f"Failed to get stocks list: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stocks list: {e}")


@router.get("/stock/current", response_model=CurrentPriceResponse, tags=["Data"])
async def get_current_price(
    symbol: str = Query(..., description="Stock symbol to get current price for")
):
    """Get current price for a specific stock."""
    try:
        # Use service method that handles all business logic
        result = await data_service.get_current_price(symbol)

        # Create stock info object
        stock_info = StockInfo(
            symbol=result["symbol"],
            name=result["stock_name"],
            current_price=result["current_price"],
        )

        return CurrentPriceResponse(
            symbol=result["symbol"],
            stock_info=stock_info,
            current_price=result["current_price"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            meta=MetaInfo(
                start_date=result["date_str"],
                end_date=result["date_str"],
                version=config.api.API_VERSION,
                message=result["message"],
                documentation="/docs",
                endpoints=["/stock/current"],
            ),
        )
    except ValueError as ve:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_detail = f"Failed to get current price for {symbol}: {str(e)}"
        api_logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/stock/historical", response_model=StockDataResponse, tags=["Data"])
async def get_historical_data(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
):
    """Get historical stock data for a specific date range."""
    try:
        # Use service method that handles all business logic
        result = await data_service.get_historical_stock_prices(
            symbol, start_date, end_date
        )

        # Convert price dictionaries to StockPrice objects
        prices = []
        for price_dict in result["prices"]:
            prices.append(
                StockPrice(
                    date=price_dict["date"],
                    open=price_dict["open"],
                    high=price_dict["high"],
                    low=price_dict["low"],
                    close=price_dict["close"],
                    volume=price_dict["volume"],
                    dividends=price_dict["dividends"],
                    stock_splits=price_dict["stock_splits"],
                )
            )

        stock_info = StockInfo(
            symbol=result["symbol"],
            name=result["stock_name"],
        )

        return StockDataResponse(
            symbol=result["symbol"],
            stock_info=stock_info,
            prices=prices,
            total_records=result["total_records"],
            meta=MetaInfo(
                start_date=result["start_date"],
                end_date=result["end_date"],
                version=config.api.API_VERSION,
                message="Historical stock data retrieved successfully",
                documentation="/docs",
                endpoints=["/stock/historical"],
            ),
        )
    except ValueError as ve:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_detail = f"Failed to get historical data for {symbol} from {start_date} to {end_date}: {str(e)}"
        api_logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/stock/recent", response_model=StockDataResponse, tags=["Data"])
async def get_recent_data(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    days_back: int = Query(..., description="Number of days back from today"),
):
    """Get recent stock data for a specific number of days back."""
    try:
        # Use service method that handles all business logic
        result = await data_service.get_recent_data(symbol, days_back)

        # Convert price dictionaries to StockPrice objects
        prices = []
        for price_dict in result["prices"]:
            prices.append(
                StockPrice(
                    date=price_dict["date"],
                    open=price_dict["open"],
                    high=price_dict["high"],
                    low=price_dict["low"],
                    close=price_dict["close"],
                    volume=price_dict["volume"],
                    dividends=price_dict["dividends"],
                    stock_splits=price_dict["stock_splits"],
                )
            )

        stock_info = StockInfo(
            symbol=result["symbol"],
            name=result["stock_name"],
        )

        return StockDataResponse(
            symbol=result["symbol"],
            stock_info=stock_info,
            prices=prices,
            total_records=result["total_records"],
            meta=MetaInfo(
                version=config.api.API_VERSION,
                message=f"Recent stock data retrieved successfully (last {days_back} days)",
                documentation="/docs",
                endpoints=["/stock/recent"],
            ),
        )
    except ValueError as ve:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_detail = (
            f"Failed to get recent data for {symbol} (last {days_back} days): {str(e)}"
        )
        api_logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/stock/from-end-date", response_model=StockDataResponse, tags=["Data"])
async def get_data_from_end_date(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    days_back: int = Query(..., description="Number of days back from end date"),
):
    """Get stock data from a specific end date going back a number of days."""
    try:
        # Use service method that handles all business logic
        result = await data_service.get_historical_stock_prices_from_end_date(
            symbol, end_date, days_back
        )

        # Convert price dictionaries to StockPrice objects
        prices = []
        for price_dict in result["prices"]:
            prices.append(
                StockPrice(
                    date=price_dict["date"],
                    open=price_dict["open"],
                    high=price_dict["high"],
                    low=price_dict["low"],
                    close=price_dict["close"],
                    volume=price_dict["volume"],
                    dividends=price_dict["dividends"],
                    stock_splits=price_dict["stock_splits"],
                )
            )

        stock_info = StockInfo(symbol=result["symbol"], name=result["stock_name"])

        return StockDataResponse(
            symbol=result["symbol"],
            stock_info=stock_info,
            prices=prices,
            total_records=result["total_records"],
            meta=MetaInfo(
                start_date=result["start_date"],
                end_date=result["end_date"],
                version=config.api.API_VERSION,
                message=f"Stock data retrieved successfully from {end_date} going back {days_back} days",
                documentation="/docs",
                endpoints=["/stock/from-end-date"],
            ),
        )
    except ValueError as ve:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_detail = f"Failed to get data from end date for {symbol} from {end_date} going back {days_back} days: {str(e)}"
        api_logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/cleanup", response_model=CleanupResponse, tags=["Data"])
async def cleanup_data(
    symbol: str = Query(None, description="Specific stock symbol to cleanup (optional)")
):
    """Clean up data for a specific symbol or all data."""
    try:
        # Perform cleanup
        result = await data_service.cleanup_data(symbol)

        return CleanupResponse(
            message=result.get("message", "Data cleanup completed successfully"),
            deleted_records=result.get("deleted_records", 0),
            symbols_affected=result.get("symbols_affected", []),
            meta=MetaInfo(
                version="1.0.0",
                message="Data cleanup operation completed",
                documentation="/docs",
                endpoints=["/cleanup"],
            ),
        )
    except Exception as e:
        error_detail = (
            f"Failed to cleanup data for symbol {symbol if symbol else 'all'}: {str(e)}"
        )
        api_logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/pre-populate", tags=["Data"])
async def pre_populate_database(
    symbols: List[str] = Query(
        None, description="List of stock symbols to pre-populate (optional)"
    ),
    days_back: int = Query(
        365, description="Number of days of historical data to fetch"
    ),
):
    """Pre-populate the database with popular stocks to improve access speed."""
    try:
        # Perform pre-population
        result = await data_service.pre_populate_popular_stocks(symbols, days_back)

        return {
            "message": f"Pre-population completed: {result['successful']} successful, {result['failed']} failed",
            "total_symbols": result["total_symbols"],
            "successful": result["successful"],
            "failed": result["failed"],
            "errors": result["errors"],
            "meta": MetaInfo(
                version="1.0.0",
                message="Database pre-population operation completed",
                documentation="/docs",
                endpoints=["/pre-populate"],
            ),
        }
    except Exception as e:
        error_detail = f"Failed to pre-populate database: {str(e)}"
        api_logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/verify-yahoo", tags=["Data"])
async def verify_yahoo_finance_data(
    symbol: str = Query(..., description="Stock symbol to verify"),
    days_back: int = Query(30, description="Number of days to look back"),
):
    """Verify if data is available in Yahoo Finance for a given symbol."""
    try:
        # Validate symbol
        data_service.validate_symbol(symbol)

        api_logger.debug(f"Verifying Yahoo Finance data for {symbol}")

        # Get verification result
        result = await data_service.verify_yahoo_finance_data(symbol, days_back)

        return result

    except ValueError as ve:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_detail = f"Failed to verify Yahoo Finance data for {symbol}: {str(e)}"
        api_logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)
