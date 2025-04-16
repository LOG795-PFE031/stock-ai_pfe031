"""
API routes for the Stock AI system.
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel

from api.schemas import (
    PredictionResponse,
    PredictionsResponse,
    NewsAnalysisResponse,
    HealthResponse,
    MetaInfo,
    ErrorResponse,
    TrainingResponse,
    TrainingStatusResponse,
    TrainingTasksResponse,
    DataUpdateResponse,
    StockDataResponse,
    NewsDataResponse,
    ModelListResponse,
    ModelMetadataResponse,
    ModelType
)
from core.config import config
from core.logging import logger
from core.utils import (
    validate_stock_symbol,
    format_prediction_response,
    get_date_range
)

# Create router
router = APIRouter()

# Initialize API logger
api_logger = logger['api']

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
            "/analyze"
        ]
    }

# Health check endpoint
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of all services."""
    try:
        # Import services from main to avoid circular imports
        from main import prediction_service, news_service, training_service, data_service, model_service
        
        # Check each service's health
        prediction_health = await prediction_service.health_check()
        news_health = await news_service.health_check()
        training_health = await training_service.health_check()
        data_health = await data_service.health_check()
        model_health = await model_service.health_check()
        
        # Create response with boolean values
        return HealthResponse(
            status="healthy" if all(h["status"] == "healthy" for h in [prediction_health, news_health, training_health, data_health, model_health]) else "unhealthy",
            components={
                "prediction_service": prediction_health["status"] == "healthy",
                "news_service": news_health["status"] == "healthy",
                "training_service": training_health["status"] == "healthy",
                "data_service": data_health["status"] == "healthy",
                "model_service": model_health["status"] == "healthy"
            },
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        api_logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

# Data collection endpoints
@router.post("/data/update/{symbol}", response_model=DataUpdateResponse, tags=["Data Services"])
async def update_stock_data(symbol: str):
    """Update stock data for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import data_service
        
        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stock symbol: {symbol}"
            )
        
        # Update data
        result = await data_service.update_data(symbol)
        
        return DataUpdateResponse(
            symbol=symbol,
            stock_data_updated=True,
            news_data_updated=True,
            timestamp=datetime.utcnow().isoformat(),
            stock_records=result["stock_data"]["rows"],
            news_articles=result["news_data"]["articles"]
        )
    except Exception as e:
        api_logger.error(f"Data update failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Data update failed: {str(e)}"
        )

@router.get("/data/stock/{symbol}", response_model=StockDataResponse, tags=["Data Services"])
async def get_stock_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get stock data for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import data_service
        
        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stock symbol: {symbol}"
            )
        
        # Get date range
        start, end = get_date_range(start_date, end_date)
        
        # Get data
        data = await data_service.get_stock_data(symbol, start, end)
        
        return StockDataResponse(
            symbol=symbol,
            data=data.to_dict(orient="records"),
            meta=MetaInfo(
                message=f"Stock data retrieved successfully for {symbol}",
                version=config.api.API_VERSION,
                documentation="https://api.example.com/docs",
                endpoints=["/api/data/stock/{symbol}"]
            ),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        api_logger.error(f"Failed to get stock data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stock data: {str(e)}"
        )

@router.get("/data/news/{symbol}", response_model=NewsDataResponse, tags=["Data Services"])
async def get_news_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get news data for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import news_service
        
        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stock symbol: {symbol}"
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
                endpoints=data["meta"]["endpoints"]
            )
        )
    except Exception as e:
        api_logger.error(f"Failed to get news data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get news data: {str(e)}"
        )

# Model management endpoints
@router.get("/models", response_model=ModelListResponse, tags=["Model Management"])
async def get_models():
    """List all available ML models."""
    try:
        # Import services from main to avoid circular imports
        from main import model_service
        
        models = await model_service.get_models()
        return ModelListResponse(models=models)
    except Exception as e:
        api_logger.error(f"Failed to get models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get models: {str(e)}"
        )

@router.get("/models/{model_id}", response_model=ModelMetadataResponse, tags=["Model Management"])
async def get_model_metadata(model_id: str):
    """Get metadata for a specific model."""
    try:
        # Import services from main to avoid circular imports
        from main import model_service
        
        metadata = await model_service.get_model_metadata(model_id)
        return ModelMetadataResponse(**metadata)
    except Exception as e:
        api_logger.error(f"Failed to get model metadata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model metadata: {str(e)}"
        )

# Prediction endpoints
@router.get("/predict/{symbol}", response_model=PredictionResponse, tags=["Prediction Services"])
async def get_next_day_prediction(
    symbol: str,
    model_type: ModelType = ModelType.LSTM
):
    """Get stock price prediction for the next day."""
    try:
        # Import services from main to avoid circular imports
        from main import prediction_service
        
        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stock symbol: {symbol}"
            )
        
        # Get prediction using the new method
        prediction = await prediction_service.get_prediction(
            symbol=symbol,
            model_type=model_type.value
        )
        
        # Check if prediction failed
        if prediction.get("status") == "error":
            error_msg = prediction.get("error", "Unknown error")
            api_logger.error(f"Prediction failed for {symbol}: {error_msg}")
            if "No Prophet model available" in error_msg:
                raise HTTPException(
                    status_code=404,
                    detail=error_msg
                )
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {error_msg}"
            )
        
        # Calculate next day's date
        current_date = datetime.now()
        next_day = current_date + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")
        
        # The prediction is already formatted by the service
        prediction["date"] = next_day_str
        prediction["symbol"] = symbol
        
        return prediction
        
    except Exception as e:
        api_logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/predict/{symbol}/historical", response_model=PredictionsResponse, tags=["Prediction Services"])
async def get_historical_predictions(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model_type: ModelType = ModelType.LSTM
):
    """Get historical predictions for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import prediction_service
        
        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stock symbol: {symbol}"
            )
        
        # Get date range
        start, end = get_date_range(start_date, end_date)
        
        # Get predictions
        predictions = await prediction_service.get_historical_predictions(
            symbol=symbol,
            start_date=start,
            end_date=end,
            model_type=model_type.value
        )
        
        # Format each prediction
        formatted_predictions = [
            format_prediction_response(
                prediction=p["predicted_price"],
                confidence=p["confidence"],
                model_type=p["model_type"],
                model_version="1.0.0",  # Default version for historical predictions
                symbol=symbol,
                date=p.get("date", None)  # Use the date from the prediction if available
            )
            for p in predictions
        ]
        
        return PredictionsResponse(
            symbol=symbol,
            predictions=formatted_predictions,
            meta=MetaInfo(
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                model_type=model_type.value
            )
        )
    except Exception as e:
        api_logger.error(f"Historical predictions failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Historical predictions failed: {str(e)}"
        )

@router.get("/predict/{symbol}/display", response_model=Dict[str, Any], tags=["Prediction Services"])
async def get_direct_display(
    symbol: str,
    model_type: ModelType = ModelType.LSTM
):
    """Get formatted prediction display for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import prediction_service, news_service
        
        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stock symbol: {symbol}"
            )
        
        # Get prediction and news analysis
        prediction = await prediction_service.get_next_day_prediction(
            symbol=symbol,
            model_type=model_type.value
        )
        news_analysis = await news_service.get_news_analysis(symbol)
        
        # Format the prediction response
        formatted_prediction = format_prediction_response(
            prediction=prediction["prediction"],
            confidence=prediction["confidence_score"],
            model_type=prediction["model_type"],
            model_version=prediction["model_version"],
            symbol=symbol
        )
        
        return {
            "prediction": formatted_prediction,
            "news_analysis": news_analysis.dict()
        }
    except Exception as e:
        api_logger.error(f"Direct display failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Direct display failed: {str(e)}"
        )

# Training endpoints
@router.post("/train/{symbol}", response_model=TrainingResponse, tags=["Training Services"])
async def train_model(
    symbol: str,
    model_type: ModelType = ModelType.LSTM,
    epochs: int = 100,
    batch_size: int = 32
):
    """Train a new model for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from main import training_service
        
        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stock symbol: {symbol}"
            )
        
        # Start training
        result = await training_service.train_model(
            symbol=symbol,
            model_type=model_type.value,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return TrainingResponse(
            symbol=symbol,
            model_type=model_type.value,
            model_version=result["model_version"],
            training_history=result["training_history"],
            metrics=result["metrics"],
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        api_logger.error(f"Training failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@router.get("/train/status/{task_id}", response_model=TrainingStatusResponse, tags=["Training Services"])
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
            status_code=500,
            detail=f"Failed to get training status: {str(e)}"
        )

@router.get("/train/tasks", response_model=TrainingTasksResponse, tags=["Training Services"])
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
            status_code=500,
            detail=f"Failed to get training tasks: {str(e)}"
        )

@router.post("/data/cleanup/{symbol}", response_model=Dict[str, Any], tags=["Data Services"])
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
                detail=f"Data cleanup failed: {result.get('message', 'Unknown error')}"
            )
        
        return result
        
    except Exception as e:
        api_logger.error(f"Data cleanup failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Data cleanup failed: {str(e)}"
        ) 