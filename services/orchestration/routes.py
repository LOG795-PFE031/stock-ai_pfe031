"""
API routes for the orchestration service (Stock AI).
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from core.config import config
from core.logging import logger
from core.utils import validate_stock_symbol, get_date_range
from .schemas import (
    MetaInfo,
    HealthResponse,
    TrainingResponse,
    PredictionResponse,
    PredictionsResponse,
    EvaluateResponse,
)

# Create router
router = APIRouter()

# Initialize API logger
api_logger = logger["orchestration"]


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
        "message": "Welcome to Stock AI Data Processing Service API",
        "version": config.api.API_VERSION,
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/metrics",
            "/trainers",
            "/train",
            "/status/{task_id}",
            "/tasks",
            "/cleanup",
        ],
    }


# Health check endpoint
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of the orchestration service."""
    try:
        # Import orchestration service from main to avoid circular imports
        from .main import orchestration_service

        # Check each service's health
        orchestration_health = await orchestration_service.health_check()

        # Create response with boolean values
        return HealthResponse(
            status=(
                "healthy"
                if orchestration_health["status"] == "healthy"
                else "unhealthy"
            ),
            components={"data_processing": orchestration_health["status"] == "healthy"},
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Health check failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(exception)}"
        ) from exception


@router.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    model_type: str = Query(..., description="Type of model to train"),
):
    """Train a new model for a symbol."""
    try:
        # Import orchestration service from main to avoid circular imports
        from .main import orchestration_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Train the model
        training_result = await orchestration_service.run_training_pipeline(
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


# Prediction endpoints
@router.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def get_next_day_prediction(
    model_type: str = Query(..., description="Type of prediction model to use"),
    symbol: str = Query(
        ..., description="Ticker symbol of the stock (e.g., AAPL, MSFT)"
    ),
):
    """Get stock price prediction for the next day."""
    try:

        from .main import orchestration_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Get prediction using the new method
        prediction = await orchestration_service.run_prediction_pipeline(
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


@router.post(
    "/predict/historical",
    response_model=PredictionsResponse,
    tags=["Predictions"],
)
async def get_historical_predictions(
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
        # Import services from main to avoid circular imports
        from .main import orchestration_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Get date range
        start, end = get_date_range(start_date, end_date)

        # Get predictions
        predictions = await orchestration_service.run_historical_prediction_pipeline(
            model_type=model_type, symbol=symbol, start_date=start, end_date=end
        )

        return predictions
    except Exception as e:
        api_logger.error(f"Historical predictions failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Historical predictions failed: {str(e)}"
        ) from e


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    tags=["Predictions"],
)
async def run_evaluation_pipeline(
    model_type: str = Query(..., description="Type of prediction model to use"),
    symbol: str = Query(
        ..., description="Ticker symbol of the stock (e.g., AAPL, MSFT)"
    ),
):
    """Get historical predictions for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from .main import orchestration_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Run evaluation pipeline
        evaluation_results = await orchestration_service.run_evaluation_pipeline(
            model_type=model_type, symbol=symbol
        )

        return evaluation_results
    except Exception as e:
        api_logger.error(f"Historical predictions failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Historical predictions failed: {str(e)}"
        ) from e
