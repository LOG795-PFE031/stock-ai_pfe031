"""
API routes for the Stock AI system.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from api.schemas import (
    ModelListMlflowResponse,
    ModelMlflowInfo,
    PredictionResponse,
    PredictionsResponse,
    HealthResponse,
    MetaInfo,
    TrainingResponse,
    TrainingTrainersResponse,
    TrainingStatusResponse,
    TrainingTasksResponse,
    StockDataResponse,
    StocksListDataResponse,
    NewsDataResponse,
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
            "/data/news",
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
        # Import services from main to avoid circular imports
        from main import (
            deployment_service,
            news_service,
            training_service,
            data_service,
            data_processing_service,
            orchestation_service,
            evaluation_service,
        )

        # Check each service's health
        deployment_health = await deployment_service.health_check()
        news_health = await news_service.health_check()
        training_health = await training_service.health_check()
        data_health = await data_service.health_check()
        preprocessing_health = await data_processing_service.health_check()
        evaluation_health = await evaluation_service.health_check()
        orchestation_health = await orchestation_service.health_check()

        # Create response with boolean values
        return HealthResponse(
            status=(
                "healthy"
                if all(
                    h["status"] == "healthy"
                    for h in [
                        deployment_health,
                        news_health,
                        training_health,
                        data_health,
                        preprocessing_health,
                        evaluation_health,
                        orchestation_health,
                    ]
                )
                else "unhealthy"
            ),
            components={
                "deployment_health": deployment_health["status"] == "healthy",
                "news_service": news_health["status"] == "healthy",
                "training_service": training_health["status"] == "healthy",
                "data_service": data_health["status"] == "healthy",
                "preprocessing_health": preprocessing_health["status"] == "healthy",
                "evaluation_health": evaluation_health["status"] == "healthy",
                "orchestation_health": orchestation_health["status"] == "healthy",
            },
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
async def get_stocks_list():
    """
    Retrieve a list of NASDAQ-100 stocks, sorted by absolute percentage change in
    descending order (top movers first)."
    """

    try:
        # Import services from main to avoid circular imports
        from main import data_service

        symbols_data = await data_service.get_nasdaq_stocks()

        return StocksListDataResponse(
            count=symbols_data["count"],
            data=symbols_data["data"],
            timestamp=datetime.now().isoformat(),
        )
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
    symbol: str = Query(..., description="Stock symbol to retrieve data for")
):
    """Get the current stock data for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import data_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        data, stock_name = await data_service.get_current_price(symbol=symbol)
        return StockDataResponse(
            symbol=symbol,
            name=stock_name,
            data=data.to_dict(orient="records"),
            meta=MetaInfo(
                message=f"Stock data retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/{symbol}"],
            ),
            timestamp=datetime.now().isoformat(),
        )

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
        # Import services from main to avoid circular imports
        from main import data_service

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

        # Check if start date if before end date
        if start_date > end_date:
            raise HTTPException(
                status_code=400, detail="start_date must be before end_date"
            )

        # Get historical data
        data, stock_name = await data_service.get_historical_stock_prices(
            symbol, start_date, end_date
        )

        return StockDataResponse(
            symbol=symbol,
            name=stock_name,
            data=data.to_dict(orient="records"),
            meta=MetaInfo(
                message=f"Stock data retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/{symbol}"],
            ),
            timestamp=datetime.now().isoformat(),
        )

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
async def get_reccent_stock_data(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    days_back: Optional[int] = Query(
        None, description="Number of days to look back", ge=1, le=10_000
    ),
):
    """Get recent stock data for a symbol (based on a number of days back)."""
    try:
        # Import services from main to avoid circular imports
        from main import data_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        if not days_back:
            raise HTTPException(
                status_code=400, detail="days_back is required for recent data"
            )

        # Get recents N trading days stock prices
        data, stock_name = await data_service.get_recent_data(
            symbol=symbol, days_back=days_back
        )

        return StockDataResponse(
            symbol=symbol,
            name=stock_name,
            data=data.to_dict(orient="records"),
            meta=MetaInfo(
                message=f"Stock data retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/{symbol}"],
            ),
            timestamp=datetime.now().isoformat(),
        )

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
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    end_date: datetime = Query(None, description="End date to retrieve the data from"),
    days_back: int = Query(
        None,
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
        # Import services from main to avoid circular imports
        from main import data_service

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

        # Get recents N trading days stock prices
        data, stock_name = await data_service.get_historical_stock_prices_from_end_date(
            symbol=symbol, days_back=days_back, end_date=end_date
        )

        return StockDataResponse(
            symbol=symbol,
            name=stock_name,
            data=data.to_dict(orient="records"),
            meta=MetaInfo(
                message=f"Stock data retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/{symbol}"],
            ),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        api_logger.error(f"Failed to get stock data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get stock data: {str(e)}"
        ) from e


@router.get("/data/news/", response_model=NewsDataResponse, tags=["Data Services"])
async def get_news_data(
    symbol: str = Query(..., description="Stock symbol to retrieve news data for"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get news data for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import news_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Get date range
        start, end = get_date_range(start_date, end_date)

        # Get data
        data = await news_service.get_news_data(symbol, start, end)

        return NewsDataResponse(
            symbol=symbol,
            articles=data["articles"],
            total_articles=data["total_articles"],
            sentiment_metrics=data["sentiment_metrics"],
            meta=MetaInfo(
                start_date=start.isoformat(),
                end_date=end.isoformat(),
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
async def get_models():
    """List all available ML models."""
    try:
        # Import services from main to avoid circular imports
        from main import deployment_service

        models = await deployment_service.list_models()
        response = ModelListMlflowResponse(
            models=models,
            total_models=len(models),
            timestamp=datetime.now().isoformat(),
        )
        return response

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
async def get_model_metadata(model_name: str):
    """Get metadata for a specific model."""
    try:
        # Import services from main to avoid circular imports
        from main import deployment_service

        metadata = await deployment_service.get_model_metadata(model_name)
        print(f"Model metadata for {model_name}: {metadata}")
        return metadata
    except Exception as e:
        api_logger.error(f"Failed to get model metadata: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model metadata: {str(e)}"
        ) from e


# Prediction endpoints
@router.get("/predict", response_model=PredictionResponse, tags=["Prediction Services"])
async def get_next_day_prediction(
    model_type: str = Query(..., description="Type of prediction model to use"),
    symbol: str = Query(
        ..., description="Ticker symbol of the stock (e.g., AAPL, MSFT)"
    ),
):
    """Get stock price prediction for the next day."""
    try:
        # Import services from main to avoid circular imports
        from main import orchestation_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Get prediction using the new method
        prediction = await orchestation_service.run_prediction_pipeline(
            model_type=model_type, symbol=symbol
        )

        if prediction.get("status") == "success":
            return prediction

        error_msg = prediction.get("error", "Unknown error")
        api_logger.error(f"Prediction failed for {symbol}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error_msg}")

    except Exception as e:
        api_logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        ) from e


@router.get(
    "/predict/historical",
    response_model=PredictionsResponse,
    tags=["Prediction Services"],
)
async def get_historical_predictions(
    model_type: str = Query(..., description="Type of prediction model to use"),
    symbol: str = Query(
        ..., description="Ticker symbol of the stock (e.g., AAPL, MSFT)"
    ),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get historical predictions for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import orchestation_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Get date range
        start, end = get_date_range(start_date, end_date)

        # Get predictions
        predictions = await orchestation_service.run_historical_prediction_pipeline(
            model_type=model_type, symbol=symbol, start_date=start, end_date=end
        )

        return predictions
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
async def get_trainers():
    """
    Retrieve the list of available training trainers.
    """
    try:
        # Import services from main to avoid circular imports
        from main import training_service

        # Get the trainers
        trainers_response = await training_service.get_trainers()

        if trainers_response.get("status") == "success":
            return TrainingTrainersResponse(
                status=trainers_response["status"],
                types=trainers_response["types"],
                count=trainers_response["count"],
                timestamp=datetime.now().isoformat(),
            )
        else:
            error_msg = trainers_response.get("error", "Unknown error")
            api_logger.error(
                f"Failed to retrieve the list of available trainers: {error_msg}"
            )
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {error_msg}"
            )

    except Exception as e:
        api_logger.error(f"Failed to retrieve the list of available trainers: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get the trainers: {str(e)}"
        ) from e


@router.post("/train", response_model=TrainingResponse, tags=["Training Services"])
async def train_model(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    model_type: str = Query(..., description="Type of model to train"),
):
    """Train a new model for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import orchestation_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Train the model
        training_result = await orchestation_service.run_training_pipeline(
            model_type=model_type, symbol=symbol
        )

        if training_result.get("status") == "success":
            return TrainingResponse(
                status=training_result["status"],
                symbol=symbol,
                model_type=model_type,
                training_results=training_result["training_results"],
                metrics=training_result["metrics"],
                deployment_results=training_result["deployment_results"],
                timestamp=datetime.now().isoformat(),
            )

        error_msg = training_result.get("error", "Unknown error")
        api_logger.error(f"Training failed for {symbol}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Training failed: {error_msg}")

    except Exception as e:
        api_logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}") from e


@router.get(
    "/train/status/{task_id}",
    response_model=TrainingStatusResponse,
    tags=["Training Services"],
)
async def get_training_status(task_id: str):
    """Check the status of a training task."""

    # TODO NOT IMPLEMENTED CORRECTLY

    try:
        # Import services from main to avoid circular imports
        from main import training_service

        status = await training_service.get_training_status(task_id)
        return TrainingStatusResponse(**status)
    except Exception as e:
        api_logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training status: {str(e)}"
        ) from e


@router.get(
    "/train/tasks", response_model=TrainingTasksResponse, tags=["Training Services"]
)
async def get_training_tasks():
    """List all training tasks."""

    # TODO NOT IMPLEMENTED CORRECTLY

    try:
        # Import services from main to avoid circular imports
        from main import training_service

        tasks = await training_service.get_training_tasks()
        return TrainingTasksResponse(tasks=tasks)
    except Exception as e:
        api_logger.error(f"Failed to get training tasks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training tasks: {str(e)}"
        ) from e


@router.post("/data/cleanup", response_model=Dict[str, Any], tags=["Data Services"])
async def cleanup_stock_data(symbol: Optional[str] = None):
    """Clean up and maintain stock data files."""
    try:
        # Import services from main to avoid circular imports
        from main import data_service

        # Clean up data
        result = await data_service.cleanup_data(symbol)

        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Data cleanup failed: {result.get('message', 'Unknown error')}",
            )

        return result

    except Exception as e:
        api_logger.error(f"Data cleanup failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Data cleanup failed: {str(e)}"
        ) from e
