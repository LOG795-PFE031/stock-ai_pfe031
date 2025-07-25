"""
API routes for the Stock AI system.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import RedirectResponse

from core.config import config
from core.logging import logger
from core.types import ProcessedDataAPI
from core.utils import validate_stock_symbol
from .schemas import (
    HealthResponse,
    MetaInfo,
    TrainingResponse,
    TrainingTrainersResponse,
    TrainingStatusResponse,
    TrainingTasksResponse,
)

# Create router
router = APIRouter()

# Initialize API logger
api_logger = logger["training"]


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
        "message": "Welcome to Stock AI Training Service API",
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
    """Check the health of all services."""
    try:
        # Import services from main to avoid circular imports
        from .main import training_service

        # Check each service's health
        training_health = await training_service.health_check()

        # Create response with boolean values
        return HealthResponse(
            status=(
                "healthy" if training_health["status"] == "healthy" else "unhealthy"
            ),
            components={"training_service": training_health["status"] == "healthy"},
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Health check failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(exception)}"
        ) from exception


# Training endpoints
@router.get(
    "/trainers",
    response_model=TrainingTrainersResponse,
    tags=["Training Services"],
)
async def get_trainers():
    """
    Retrieve the list of available training trainers.
    """
    try:
        # Import services from main to avoid circular imports
        from .main import training_service

        # Get the trainers
        trainers_response = await training_service.get_trainers()

        # If response successful return the trainers
        if trainers_response.get("status") == "success":
            return TrainingTrainersResponse(
                status=trainers_response["status"],
                types=trainers_response["types"],
                count=trainers_response["count"],
                timestamp=datetime.now().isoformat(),
            )

        # Else return error
        error_msg = trainers_response.get("error", "Unknown error")
        api_logger.error(
            f"Failed to retrieve the list of available trainers: {error_msg}"
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error_msg}")

    except Exception as exception:
        api_logger.error(
            f"Failed to retrieve the list of available trainers: {str(exception)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get the trainers: {str(exception)}"
        ) from exception


@router.post("/train", response_model=TrainingResponse, tags=["Training Services"])
async def train_model(
    symbol: str = Query(..., description="Stock symbol to retrieve data for"),
    model_type: str = Query(..., description="Type of model to train"),
    preprocessed_data: ProcessedDataAPI = Body(
        ..., description="Data to train the model"
    ),
):
    """Train a new model for a symbol."""
    try:
        # Import services from main to avoid circular imports
        from .main import training_service

        # Validate symbol
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=400, detail=f"Invalid stock symbol: {symbol}"
            )

        # Train the model
        training_result = await training_service.train_model(
            model_type=model_type, symbol=symbol, data=preprocessed_data.data
        )

        if training_result.get("status") == "success":
            return TrainingResponse(
                status=training_result["status"],
                symbol=symbol,
                model_type=model_type,
                run_id=training_result["run_id"],
                run_info=training_result["run_info"],
                training_history=training_result["training_history"],
                timestamp=training_result["timestamp"],
            )

        error_msg = training_result.get("error", "Unknown error")
        api_logger.error(f"Training failed for {symbol}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Training failed: {error_msg}")

    except Exception as exception:
        api_logger.error(f"Training failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Training failed: {str(exception)}"
        ) from exception


@router.get(
    "/status/{task_id}",
    response_model=TrainingStatusResponse,
    tags=["Training Services"],
)
async def get_training_status(
    symbol: str = Query(..., description="Stock symbol"),
    model_type: str = Query(..., description="Type of model used for training"),
):
    """Check the status of a training task."""

    try:
        # Import services from main to avoid circular imports
        from .main import training_service

        status = await training_service.get_training_status(symbol, model_type)
        return TrainingStatusResponse(**status)
    except Exception as exception:
        api_logger.error(f"Failed to get training status: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training status: {str(exception)}"
        ) from exception


@router.get("/tasks", response_model=TrainingTasksResponse, tags=["Training Services"])
async def get_training_tasks():
    """List all training tasks."""

    try:
        # Import services from main to avoid circular imports
        from .main import training_service

        tasks = await training_service.get_active_training_tasks()
        return TrainingTasksResponse(
            tasks=tasks,
            total_tasks=len(tasks),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Failed to get training tasks: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training tasks: {str(exception)}"
        ) from exception


@router.post("/cleanup", response_model=Dict[str, Any], tags=["Data Services"])
async def cleanup_training_service():
    """Clean up the training service."""
    try:
        # Import services from main to avoid circular imports
        from .main import training_service

        # Clean up training service
        result = await training_service.cleanup()

        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Training service cleanup failed: {result.get('message', 'Unknown error')}",
            )

        return result

    except Exception as exception:
        api_logger.error(f"Training service cleanup failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Training service cleanup failed: {str(exception)}"
        ) from exception
