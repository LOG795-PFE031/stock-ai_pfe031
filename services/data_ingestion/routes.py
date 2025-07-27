from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from datetime import datetime
from typing import List
from .data_service import DataService
from .schemas import (
    StockDataResponse, 
    CurrentPriceResponse, 
    CleanupResponse, 
    MetaInfo, 
    HealthResponse,
    StockInfo,
    StockPrice
)
from core.config import config
from core.utils import get_date_range
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
        "endpoints": ["/stocks", "/stock/current", "/stock/historical", "/stock/recent", "/stock/from-end-date", "/cleanup"],
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
            status=(
                "healthy" if data_health["status"] == "healthy" else "unhealthy"
            ),
            components={"data_service": data_health["status"] == "healthy"},
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Health check failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(exception)}"
        ) from exception

@router.get("/stocks", response_model=List[StockInfo], tags=["Data"])
async def get_stocks():
    """Get list of available stocks."""
    try:
        # This could be enhanced to return actual available stocks from database
        # For now, returning a sample list
        stocks = [
            StockInfo(symbol="AAPL", name="Apple Inc."),
            StockInfo(symbol="MSFT", name="Microsoft Corporation"),
            StockInfo(symbol="GOOGL", name="Alphabet Inc."),
            StockInfo(symbol="AMZN", name="Amazon.com Inc."),
            StockInfo(symbol="TSLA", name="Tesla Inc.")
        ]
        return stocks
    except Exception as e:
        api_logger.error(f"Failed to get stocks list: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stocks list: {e}")

@router.get("/stock/current", response_model=CurrentPriceResponse, tags=["Data"])
async def get_current_price(
    symbol: str = Query(..., description="Stock symbol to get current price for")
):
    """Get current price for a specific stock."""
    try:
        # Validate symbol
        if not symbol.isalnum():
            raise HTTPException(status_code=400, detail=f"Invalid stock symbol: {symbol}")
        
        # Get current price data
        current_price_df, _ = await data_service.get_current_price(symbol)
        
        # Extract current price from DataFrame
        if not current_price_df.empty:
            current_price = float(current_price_df.iloc[0]['Close'])
            # Calculate change percent
            change_percent = await data_service.calculate_change_percent(symbol)
        else:
            current_price = 0.0
            change_percent = None
        
        now_iso = datetime.utcnow().isoformat()
        return CurrentPriceResponse(
            symbol=symbol,
            current_price=current_price,
            change_percent=change_percent,  
            timestamp=now_iso,
            meta=MetaInfo(
                start_date=now_iso,
                end_date=now_iso,
                version=config.api.API_VERSION,
                message="Current price retrieved successfully",
                documentation="/docs",
                endpoints=["/stock/current"],
            ),
        )
    except Exception as e:
        error_detail = f"Failed to get current price for {symbol}: {str(e)}"
        api_logger.error(error_detail)
        api_logger.error(f"Full error details: {repr(e)}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/stock/historical", response_model=StockDataResponse, tags=["Data"])
async def get_historical_data(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """Get historical stock data for a specific date range."""
    try:
        # Validate symbol
        if not symbol.isalnum():
            raise HTTPException(status_code=400, detail=f"Invalid stock symbol: {symbol}")
        
        # Get date range
        start, end = get_date_range(start_date, end_date)
        
        # Get historical data
        data, stock_name = await data_service.get_historical_stock_prices(symbol, start, end)
        
        # Get current price data
        current_price_df, _ = await data_service.get_current_price(symbol)
        
        # Extract current price from DataFrame
        if not current_price_df.empty:
            current_price = float(current_price_df.iloc[0]['Close'])
            # Calculate change percent
            change_percent = await data_service.calculate_change_percent(symbol)
        else:
            current_price = None
            change_percent = None
        
        # Convert DataFrame to list of StockPrice objects
        prices = []
        for _, row in data.iterrows():
            # Handle date formatting safely
            if hasattr(row.name, 'strftime'):
                date_str = row.name.strftime('%Y-%m-%d')
            elif 'Date' in row and hasattr(row['Date'], 'strftime'):
                date_str = row['Date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row.name)
                
            prices.append(StockPrice(
                date=date_str,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                adj_close=float(row['Adj Close']) if 'Adj Close' in row else None
            ))
        
        stock_info = StockInfo(
            symbol=symbol,
            name=stock_name,
            current_price=current_price,
            change_percent=change_percent  
        )
        
        return StockDataResponse(
            symbol=symbol,
            stock_info=stock_info,
            prices=prices,
            total_records=len(prices),
            meta=MetaInfo(
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                version=config.api.API_VERSION,
                message="Historical stock data retrieved successfully",
                documentation="/docs",
                endpoints=["/stock/historical"],
            ),
        )
    except Exception as e:
        error_detail = f"Failed to get historical data for {symbol} from {start_date} to {end_date}: {str(e)}"
        api_logger.error(error_detail)
        api_logger.error(f"Full error details: {repr(e)}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/stock/recent", response_model=StockDataResponse, tags=["Data"])
async def get_recent_data(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    days_back: int = Query(..., description="Number of days back from today")
):
    """Get recent stock data for a specific number of days back."""
    try:
        # Validate symbol
        if not symbol.isalnum():
            raise HTTPException(status_code=400, detail=f"Invalid stock symbol: {symbol}")
        
        # Validate days_back
        if days_back <= 0 or days_back > 365:
            raise HTTPException(status_code=400, detail="days_back must be between 1 and 365")
        
        # Get recent data
        data, stock_name = await data_service.get_recent_data(symbol, days_back)
        
        # Get current price data
        current_price_df, _ = await data_service.get_current_price(symbol)
        
        # Extract current price from DataFrame
        if not current_price_df.empty:
            current_price = float(current_price_df.iloc[0]['Close'])
            # Calculate change percent
            change_percent = await data_service.calculate_change_percent(symbol)
        else:
            current_price = None
            change_percent = None
        
        # Convert DataFrame to list of StockPrice objects
        prices = []
        for _, row in data.iterrows():
            # Handle date formatting safely
            if hasattr(row.name, 'strftime'):
                date_str = row.name.strftime('%Y-%m-%d')
            elif 'Date' in row and hasattr(row['Date'], 'strftime'):
                date_str = row['Date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row.name)
                
            prices.append(StockPrice(
                date=date_str,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                adj_close=float(row['Adj Close']) if 'Adj Close' in row else None
            ))
        
        stock_info = StockInfo(
            symbol=symbol,
            name=stock_name,
            current_price=current_price,
            change_percent=change_percent  # Now using the calculated value
        )
        
        return StockDataResponse(
            symbol=symbol,
            stock_info=stock_info,
            prices=prices,
            total_records=len(prices),
            meta=MetaInfo(
                version=config.api.API_VERSION,
                message=f"Recent stock data retrieved successfully (last {days_back} days)",
                documentation="/docs",
                endpoints=["/stock/recent"],
            ),
        )
    except Exception as e:
        error_detail = f"Failed to get recent data for {symbol} (last {days_back} days): {str(e)}"
        api_logger.error(error_detail)
        api_logger.error(f"Full error details: {repr(e)}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/stock/from-end-date", response_model=StockDataResponse, tags=["Data"])
async def get_data_from_end_date(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    days_back: int = Query(..., description="Number of days back from end date")
):
    """Get stock data from a specific end date going back a number of days."""
    try:
        # Validate symbol
        if not symbol.isalnum():
            raise HTTPException(status_code=400, detail=f"Invalid stock symbol: {symbol}")
        
        # Validate days_back
        if days_back <= 0 or days_back > 365:
            raise HTTPException(status_code=400, detail="days_back must be between 1 and 365")
        
        # Parse end date
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
        
        # Get data from end date
        data, stock_name = await data_service.get_historical_stock_prices_from_end_date(symbol, end, days_back)
        
        # Get current price data
        current_price_df, _ = await data_service.get_current_price(symbol)
        
        # Extract current price from DataFrame
        if not current_price_df.empty:
            current_price = float(current_price_df.iloc[0]['Close'])
            # Calculate change percent
            change_percent = await data_service.calculate_change_percent(symbol)
        else:
            current_price = None
            change_percent = None
        
        # Get the oldest date from the data
        start_date = None
        if not data.empty:
            # Sort data by Date if not already sorted
            if 'Date' in data.columns:
                data = data.sort_values('Date')
                start_date = data.iloc[0]['Date']
            else:
                # Assuming the index is the date
                data = data.sort_index()
                start_date = data.index[0]
        
        # Format start_date for the response
        start_date_iso = start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date)
        
        # Convert DataFrame to list of StockPrice objects
        prices = []
        for _, row in data.iterrows():
            # Handle date formatting safely
            if hasattr(row.name, 'strftime'):
                date_str = row.name.strftime('%Y-%m-%d')
            elif 'Date' in row and hasattr(row['Date'], 'strftime'):
                date_str = row['Date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row.name)
                
            prices.append(StockPrice(
                date=date_str,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                adj_close=float(row['Adj Close']) if 'Adj Close' in row else None
            ))
        
        stock_info = StockInfo(
            symbol=symbol,
            name=stock_name,
            current_price=current_price,
            change_percent=change_percent
        )
        
        return StockDataResponse(
            symbol=symbol,
            stock_info=stock_info,
            prices=prices,
            total_records=len(prices),
            meta=MetaInfo(
                start_date=start_date_iso,
                end_date=end.isoformat(),
                version=config.api.API_VERSION,
                message=f"Stock data retrieved successfully from {end_date} going back {days_back} days",
                documentation="/docs",
                endpoints=["/stock/from-end-date"],
            ),
        )
    except Exception as e:
        error_detail = f"Failed to get data from end date for {symbol} from {end_date} going back {days_back} days: {str(e)}"
        api_logger.error(error_detail)
        api_logger.error(f"Full error details: {repr(e)}")
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
        error_detail = f"Failed to cleanup data for symbol {symbol if symbol else 'all'}: {str(e)}"
        api_logger.error(error_detail)
        api_logger.error(f"Full error details: {repr(e)}")
        raise HTTPException(status_code=500, detail=error_detail)
