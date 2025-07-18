"""
API routes for the Stock AI system.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from typing import Dict, Any, Optional
from datetime import datetime
from core.utils import get_next_trading_day
from monitoring.prometheus_metrics import prediction_time_seconds
import time

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
    DataUpdateResponse,
    StockDataResponse,
    StocksListDataResponse,
    NewsDataResponse,
    ModelListResponse,
    ModelMetadataResponse,
)
from core.config import config
from core.logging import logger
from core.utils import validate_stock_symbol, format_prediction_response, get_date_range

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
            preprocessing_service,
            evaluation_service,
        )

        # Check each service's health
        deployment_health = await deployment_service.health_check()
        news_health = await news_service.health_check()
        training_health = await training_service.health_check()
        data_health = await data_service.health_check()
        preprocessing_health = await preprocessing_service.health_check()
        evaluation_health = await evaluation_service.health_check()

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
            },
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        api_logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


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
        )


@router.get(
    "/data/stock/{symbol}/current",
    response_model=StockDataResponse,
    tags=["Data Services"],
)
async def get_current_stock_data(
    symbol: str,
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
        )


@router.get(
    "/data/stock/{symbol}/historical",
    response_model=StockDataResponse,
    tags=["Data Services"],
)
async def get_historical_stock_data(
    symbol: str,
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
        )


@router.get(
    "/data/stock/{symbol}/recent",
    response_model=StockDataResponse,
    tags=["Data Services"],
)
async def get_reccent_stock_data(
    symbol: str,
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
        )


@router.get(
    "/data/news/{symbol}", response_model=NewsDataResponse, tags=["Data Services"]
)
async def get_news_data(
    symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None
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
        )


# Model management endpoints
@router.get("/models", response_model=ModelListMlflowResponse, tags=["Model Management"])
async def get_models():
    """List all available ML models."""
    try:
        # Import services from main to avoid circular imports
        from main import deployment_service

        models = await deployment_service.list_models()
        response = ModelListMlflowResponse(models=models,
                                        total_models=len(models),
                                        timestamp=datetime.now().isoformat())
        return response

    except Exception as e:
        api_logger.error(f"Failed to get models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


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
        )


# Prediction endpoints
@router.get(
    "/predict/{symbol}", response_model=PredictionResponse, tags=["Prediction Services"]
)
async def get_next_day_prediction(symbol: str, model_type: str = "lstm"):
    """Get stock price prediction for the next day."""
    try:
        # Import services from main to avoid circular imports
        from main import orchestation_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        start_time = time.time()

        # Get prediction using the new method
        prediction = await orchestation_service.run_prediction_pipeline(
            model_type=model_type, symbol=symbol
        )

        elapsed = time.time() - start_time

        if prediction.get("status") == "success":
            # Observe latency in seconds
            prediction_time_seconds.labels(
                model_type=model_type, symbol=symbol
            ).observe(elapsed)

            return prediction

        else:
            error_msg = prediction.get("error", "Unknown error")
            api_logger.error(f"Prediction failed for {symbol}: {error_msg}")
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {error_msg}"
            )

    except Exception as e:
        api_logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get(
    "/predict/{symbol}/historical",
    response_model=PredictionsResponse,
    tags=["Prediction Services"],
)
async def get_historical_predictions(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model_type: str = "lstm",
):
    """Get historical predictions for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import prediction_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Get date range
        start, end = get_date_range(start_date, end_date)

        # Get predictions
        predictions = await prediction_service.get_historical_predictions(
            symbol=symbol, start_date=start, end_date=end, model_type=model_type
        )

        # Format each prediction
        formatted_predictions = [
            format_prediction_response(
                prediction=p["predicted_price"],
                confidence=p["confidence"],
                model_type=p["model_type"],
                model_version="1.0.0",  # Default version for historical predictions
                symbol=symbol,
                date=p.get(
                    "date", None
                ),  # Use the date from the prediction if available
            )
            for p in predictions
        ]

        return PredictionsResponse(
            symbol=symbol,
            predictions=formatted_predictions,
            meta=MetaInfo(
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                model_type=model_type,
            ),
        )
    except Exception as e:
        api_logger.error(f"Historical predictions failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Historical predictions failed: {str(e)}"
        )


@router.get(
    "/predict/{symbol}/display",
    response_model=Dict[str, Any],
    tags=["Prediction Services"],
)
async def get_direct_display(symbol: str, model_type: str = "lstm"):
    """Get formatted prediction display for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import prediction_service, news_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Get prediction and news analysis
        prediction = await prediction_service.get_next_day_prediction(
            symbol=symbol, model_type=model_type
        )
        news_analysis = await news_service.get_news_analysis(symbol)

        # Format the prediction response
        formatted_prediction = format_prediction_response(
            prediction=prediction["prediction"],
            confidence=prediction["confidence_score"],
            model_type=prediction["model_type"],
            model_version=prediction["model_version"],
            symbol=symbol,
        )

        return {
            "prediction": formatted_prediction,
            "news_analysis": news_analysis.dict(),
        }
    except Exception as e:
        api_logger.error(f"Direct display failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Direct display failed: {str(e)}")


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
        )


@router.post(
    "/train/{symbol}", response_model=TrainingResponse, tags=["Training Services"]
)
async def train_model(
    symbol: str,
    model_type: str = "lstm",
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
        else:
            error_msg = training_result.get("error", "Unknown error")
            api_logger.error(f"Training failed for {symbol}: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Training failed: {error_msg}")

    except Exception as e:
        api_logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get(
    "/train/status/{task_id}",
    response_model=TrainingStatusResponse,
    tags=["Training Services"],
)
async def get_training_status(task_id: str):
    """Check the status of a training task."""
    try:
        # Import services from main to avoid circular imports
        from main import training_service

        status = await training_service.get_training_status(task_id)
        return TrainingStatusResponse(**status)
    except Exception as e:
        api_logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training status: {str(e)}"
        )


@router.get(
    "/train/tasks", response_model=TrainingTasksResponse, tags=["Training Services"]
)
async def get_training_tasks():
    """List all training tasks."""
    try:
        # Import services from main to avoid circular imports
        from main import training_service

        tasks = await training_service.get_training_tasks()
        return TrainingTasksResponse(tasks=tasks)
    except Exception as e:
        api_logger.error(f"Failed to get training tasks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training tasks: {str(e)}"
        )


@router.post(
    "/data/cleanup/{symbol}", response_model=Dict[str, Any], tags=["Data Services"]
)
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
        raise HTTPException(status_code=500, detail=f"Data cleanup failed: {str(e)}")
