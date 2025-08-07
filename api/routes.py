"""
API routes for the Stock AI system.
"""

from typing import Dict, Any, Optional
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from api.schemas import (
    ModelListMlflowResponse,
    ModelMlflowInfo,
    PredictionResponse,
    PredictionsResponse,
    HealthResponse,
    MetaInfo,
    StockDataResponse,
    StocksListDataResponse,
    NewsDataResponse,
    TrainingTrainersResponse,
    TrainingResponse,
)


from core.config import config
from core.logging import logger
from core.utils import validate_stock_symbol, get_date_range

# Create router
router = APIRouter()

# Initialize API logger
api_logger = logger["api"]


# Root endpoint
@router.get("/", response_class=RedirectResponse, tags=["System"])
async def root():
    """Redirect to API documentation."""
    return "/docs"


# API welcome message
@router.get("/welcome", response_model=MetaInfo, tags=["System"])
async def api_welcome():
    """Get API welcome message and information."""
    return {
        "message": "Welcome to Stock AI API",
        "version": config.api.API_VERSION,
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/data/update",
            "/data/stock",
            "/models",
            "/predict",
            "/analyze",
        ],
    }


# Health check endpoint
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of all services."""
    try:
        return HealthResponse(
            status="healthy",
            components={},
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        api_logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(e)}"
        ) from e


# Data collection endpoints
@router.get(
    "/data/stocks", response_model=StocksListDataResponse, tags=["Data Services"]
)
async def get_stocks_list(request: Request):
    """
    Retrieve a list of NASDAQ-100 stocks, sorted by absolute percentage change in
    descending order (top movers first).
    """
    try:
        client = request.app.state.httpx_client

        # Call the data ingestion service
        data_service_url = f"http://{config.data.HOST}:{config.data.PORT}/data/stocks"

        response = await client.get(data_service_url)
        response.raise_for_status()
        stocks_response = response.json()

        # The data ingestion service already returns a StocksListDataResponse structure
        # So we can return it directly, just updating the timestamp
        return StocksListDataResponse(
            count=stocks_response["count"],
            data=stocks_response["data"],
            timestamp=datetime.now().isoformat(),
        )
    except httpx.HTTPError as e:
        api_logger.error(f"HTTP error calling data service: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock list: {str(e)}"
        ) from e
    except Exception as e:
        api_logger.error(f"Failed to get stock list: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock list: {str(e)}"
        ) from e


@router.get(
    "/data/stock/current",
    response_model=StockDataResponse,
    tags=["Data Services"],
)
async def get_current_stock_data(
    request: Request,
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
):
    """Get the current stock data for a symbol."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        logger["main"].info(f"Fetching current stock data for {symbol}...")

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Call the data ingestion service
        data_service_url = (
            f"http://{config.data.HOST}:{config.data.PORT}/data/stock/current"
        )

        response = await client.get(data_service_url, params={"symbol": symbol})
        response.raise_for_status()
        stock_data = response.json()

        logger["main"].info(
            f"✅ Successfully retrieved stock data for symbol: {symbol}"
        )

        # The data ingestion service returns a CurrentPriceResponse, extract the data
        # Ensure 'data' is a list of dicts as expected by StockDataResponse
        current_price = stock_data["prices"]
        if isinstance(current_price, dict):
            current_price = [current_price]

        return StockDataResponse(
            symbol=stock_data["stock_info"]["symbol"],
            name=stock_data["stock_info"]["name"],
            data=current_price,
            meta=MetaInfo(
                message=f"Current stock data retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/current"],
            ),
            timestamp=datetime.now().isoformat(),
        )

    except httpx.HTTPError as e:
        api_logger.error(f"HTTP error calling data service: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e
    except Exception as e:
        api_logger.error(f"Failed to get stock data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e


@router.get(
    "/data/stock/historical",
    response_model=StockDataResponse,
    tags=["Data Services"],
)
async def get_historical_stock_data(
    request: Request,
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    start_date: Optional[datetime] = Query(
        None, description="Start date for historical data"
    ),
    end_date: Optional[datetime] = Query(
        None, description="End date for historical data (defaults to today)"
    ),
):
    """Get historical stock data for a symbol."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        if not start_date:
            raise HTTPException(
                status_code=400, detail="start_date is required for historical data"
            )
        end_date = end_date or datetime.now()

        # Check if start date is before end date
        if start_date > end_date:
            raise HTTPException(
                status_code=400, detail="start_date must be before end_date"
            )

        # Call the data ingestion service
        data_service_url = (
            f"http://{config.data.HOST}:{config.data.PORT}/data/stock/historical"
        )
        params = {
            "symbol": symbol,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }

        response = await client.get(data_service_url, params=params)
        response.raise_for_status()
        stock_data = response.json()

        # The data ingestion service returns a StockDataResponse structure
        return StockDataResponse(
            symbol=stock_data["stock_info"]["symbol"],
            name=stock_data["stock_info"]["name"],
            data=stock_data["prices"],
            meta=MetaInfo(
                message=f"Historical stock data retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/historical"],
            ),
            timestamp=datetime.now().isoformat(),
        )

    except httpx.HTTPError as e:
        api_logger.error(f"HTTP error calling data service: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e
    except Exception as e:
        api_logger.error(f"Failed to get stock data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e


@router.get(
    "/data/stock/recent",
    response_model=StockDataResponse,
    tags=["Data Services"],
)
async def get_recent_stock_data(
    request: Request,
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    days_back: int = Query(
        config.data.LOOKBACK_PERIOD_DAYS,
        description="Number of days to look back (default: 365)",
        ge=1,
        le=10_000,
    ),
):
    """Get recent stock data for a symbol (based on a number of days back)."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Call the data ingestion service
        data_service_url = (
            f"http://{config.data.HOST}:{config.data.PORT}/data/stock/recent"
        )
        params = {"symbol": symbol, "days_back": days_back}

        response = await client.get(data_service_url, params=params)
        response.raise_for_status()
        stock_data = response.json()

        # The data ingestion service returns a StockDataResponse structure
        return StockDataResponse(
            symbol=stock_data["stock_info"]["symbol"],
            name=stock_data["stock_info"]["name"],
            data=stock_data["prices"],
            meta=MetaInfo(
                message=f"Recent stock data retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/recent"],
            ),
            timestamp=datetime.now().isoformat(),
        )

    except httpx.HTTPError as e:
        api_logger.error(f"HTTP error calling data service: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e
    except Exception as e:
        api_logger.error(f"Failed to get stock data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e


@router.get(
    "/data/stock/from-end-date",
    response_model=StockDataResponse,
    tags=["Data Services"],
)
async def get_historical_stock_prices_from_end_date(
    request: Request,
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    end_date: datetime = Query(..., description="End date to retrieve the data from"),
    days_back: int = Query(
        ...,
        description="Number of days to look back from the end date",
        ge=1,
        le=10_000,
    ),
):
    """
    Retrieve stock prices for a symbol from a specified end date, looking back a given number
    of days.
    """
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        if not end_date:
            raise HTTPException(
                status_code=400,
                detail="end_date is required for get_historical_stock_prices_from_end_date",
            )

        if not days_back:
            raise HTTPException(
                status_code=400,
                detail="days_back is required for get_historical_stock_prices_from_end_date",
            )

        # Call the data ingestion service
        data_service_url = (
            f"http://{config.data.HOST}:{config.data.PORT}/data/stock/from-end-date"
        )
        params = {
            "symbol": symbol,
            "end_date": end_date.strftime("%Y-%m-%d"),
            "days_back": days_back,
        }

        response = await client.get(data_service_url, params=params)
        response.raise_for_status()
        stock_data = response.json()

        # The data ingestion service returns a StockDataResponse structure
        return StockDataResponse(
            symbol=stock_data["stock_info"]["symbol"],
            name=stock_data["stock_info"]["name"],
            data=stock_data["prices"],
            meta=MetaInfo(
                message=f"Stock data from end date retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/from-end-date"],
            ),
            timestamp=datetime.now().isoformat(),
        )

    except httpx.HTTPError as e:
        api_logger.error(f"HTTP error calling data service: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e
    except Exception as e:
        api_logger.error(f"Failed to get stock data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e


@router.get("/data/news", response_model=NewsDataResponse, tags=["News Services"])
async def get_news_data(
    request: Request,
    symbol: str = Query(..., description="Stock symbol to retrieve news data for"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get news data for a symbol."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # Import services from main to avoid circular imports
        url = f"http://{config.news_service.HOST}:{config.news_service.PORT}/data/news"

        # Build params dict, excluding None values
        params = {"symbol": symbol}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        return NewsDataResponse(
            symbol=data["symbol"],
            articles=data["articles"],
            total_articles=data["total_articles"],
            sentiment_metrics=data["sentiment_metrics"],
            meta=MetaInfo(
                start_date=data["meta"].get("start_date"),
                end_date=data["meta"].get("end_date"),
                version=data["meta"]["version"],
                message=data["meta"]["message"],
                documentation=data["meta"]["documentation"],
                endpoints=data["meta"]["endpoints"],
            ),
        )
    except Exception as e:
        api_logger.error(f"Failed to get news data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get news data: {str(e)}"
        ) from e


# Model management endpoints
@router.get(
    "/models", response_model=ModelListMlflowResponse, tags=["Model Management"]
)
async def get_models(request: Request):
    """List all available ML models."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/models"

        response = await client.get(url)
        response.raise_for_status()
        models_response = response.json()

        return ModelListMlflowResponse(**models_response)

    except Exception as e:
        api_logger.error(f"Failed to get models: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get models: {str(e)}"
        ) from e


@router.get(
    "/models/{model_name}",
    response_model=ModelMlflowInfo,
    tags=["Model Management"],
)
async def get_model_metadata(request: Request, model_name: str):
    """Get metadata for a specific model."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/models/{model_name}"

        r = await client.get(url)
        r.raise_for_status()
        metadata = r.json()

        print(f"Model metadata for {model_name}: {metadata}")
        return metadata
    except Exception as e:
        api_logger.error(f"Failed to get model metadata: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model metadata: {str(e)}"
        ) from e


# Prediction endpoints
@router.post(
    "/predict", response_model=PredictionResponse, tags=["Prediction Services"]
)
async def get_next_day_prediction(
    request: Request,
    model_type: str = Query(..., description="Type of prediction model to use"),
    symbol: str = Query(
        ..., description="Ticker symbol of the stock (e.g., AAPL, MSFT)"
    ),
):
    """Get stock price prediction for the next day."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # URL to the endpoint fro prediction
        url = f"http://{config.orchestration_service.HOST}:{config.orchestration_service.PORT}/orchestration/predict"

        # Define the query parameters
        params = {
            "symbol": symbol,
            "model_type": model_type,
        }

        api_logger.info(
            "Sending POST request to the orchestration service for prediction"
        )

        # Send POST request to FastAPI endpoint
        response = await client.post(url, params=params)

        # Check if the response is successful
        response.raise_for_status()

        api_logger.info(
            "Successfully received response from the orchestration service for prediction"
        )

        # Return the response as is
        return response.json()

    except httpx.HTTPStatusError as e:
        # Server responded but with 4xx or 5xx status
        api_logger.error(
            f"HTTP error from orchestration service: {e.response.status_code} - {e.response.text}"
        )
        raise HTTPException(
            status_code=e.response.status_code, detail=e.response.text
        ) from e

    except Exception as e:
        api_logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        ) from e


@router.post(
    "/predict/historical",
    response_model=PredictionsResponse,
    tags=["Prediction Services"],
)
async def get_historical_predictions(
    request: Request,
    model_type: str = Query(..., description="Type of prediction model to use"),
    symbol: str = Query(
        ..., description="Ticker symbol of the stock (e.g., AAPL, MSFT)"
    ),
    start_date: Optional[str] = Query(
        None, description="Start date for prediction (optional)"
    ),
    end_date: Optional[str] = Query(
        None, description="End date for prediction (optional)"
    ),
):
    """Get historical predictions for a symbol."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # URL to the endpoint fro prediction
        url = f"http://{config.orchestration_service.HOST}:{config.orchestration_service.PORT}/orchestration/predict/historical"

        # Define the query parameters
        params = {
            "symbol": symbol,
            "model_type": model_type,
            "start_date": start_date,
            "end_date": end_date,
        }

        api_logger.info(
            "Sending POST request to the orchestration service for historical prediction"
        )

        # Send POST request to FastAPI endpoint
        response = await client.post(url, params=params)

        # Check if the response is successful
        response.raise_for_status()

        api_logger.info(
            "Successfully received response from the orchestration service for historical prediction"
        )

        # Return the response as is
        return response.json()

    except Exception as e:
        api_logger.error(f"Historical predictions failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Historical predictions failed: {str(e)}"
        ) from e


# Training endpoints
@router.get(
    "/train/trainers",
    response_model=TrainingTrainersResponse,
    tags=["Training Services"],
)
async def get_trainers(request: Request):
    """
    Retrieve the list of available training trainers.
    """
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # URL to the endpoint to fetch the list of available trainers
        url = f"http://{config.training_service.HOST}:{config.training_service.PORT}/training/trainers"

        # Get the trainers
        response = await client.get(url)

        # Check if the response is successful
        response.raise_for_status()

        trainers_response = response.json()
        if trainers_response.get("status") == "success":
            return TrainingTrainersResponse(
                status=trainers_response["status"],
                types=trainers_response["types"],
                count=trainers_response["count"],
                timestamp=datetime.now().isoformat(),
            )

        error_msg = trainers_response.get("error", "Unknown error")
        api_logger.error(
            f"Failed to retrieve the list of available trainers: {error_msg}"
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error_msg}")

    except Exception as e:
        api_logger.error(f"Failed to retrieve the list of available trainers: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get the trainers: {str(e)}"
        ) from e


@router.post("/train", response_model=TrainingResponse, tags=["Training Services"])
async def train_model(
    request: Request,
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    model_type: str = Query(..., description="Type of model to train"),
):
    """Train a new model for a symbol."""
    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # URL to the endpoint to train the model
        url = f"http://{config.orchestration_service.HOST}:{config.orchestration_service.PORT}/orchestration/train"

        # Define the query parameters
        params = {
            "symbol": symbol,
            "model_type": model_type,
        }

        api_logger.info(
            "Sending POST request to the orchestration service for training"
        )

        # Send POST request to FastAPI endpoint
        response = await client.post(url, params=params)

        # Check if the response is successful
        response.raise_for_status()

        api_logger.info(
            "Successfully received response from the orchestration service for training"
        )

        # Return the response as is
        return response.json()

    except Exception as e:
        api_logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}") from e


@router.post("/data/cleanup", response_model=Dict[str, Any], tags=["Data Services"])
async def cleanup_stock_data(request: Request, symbol: Optional[str] = None):
    """Clean up and maintain stock data files."""

    try:
        # Get the httpx client
        client = request.app.state.httpx_client

        # Call the data ingestion service
        data_service_url = f"http://{config.data.HOST}:{config.data.PORT}/data/cleanup"
        params = {"symbol": symbol} if symbol else {}

        response = await client.post(data_service_url, params=params)
        response.raise_for_status()
        cleanup_result = response.json()

        # The data ingestion service returns a CleanupResponse structure
        # Return it with updated timestamp
        return {
            "status": cleanup_result.get("status", "completed"),
            "message": cleanup_result.get("message", "Data cleanup completed"),
            "files_processed": cleanup_result.get("files_processed", 0),
            "files_deleted": cleanup_result.get("files_deleted", 0),
            "symbol": cleanup_result.get("symbol"),
            "timestamp": datetime.now().isoformat(),
        }

    except httpx.HTTPError as e:
        api_logger.error(f"HTTP error calling data service: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Data cleanup failed: {str(e)}"
        ) from e
    except Exception as e:
        api_logger.error(f"Data cleanup failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Data cleanup failed: {str(e)}"
        ) from e
