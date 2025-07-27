"""
API routes for the data processing service (Stock AI).
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import RedirectResponse

from core.config import config
from core.logging import logger
from .schemas import (
    HealthResponse,
    MetaInfo,
    PredictionInput,
    ProcessedData,
    RawStockDataInput,
    ScalerPromotionResponse,
)

# Create router
router = APIRouter()

# Initialize API logger
api_logger = logger["data_processing"]


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
    """Check the health of the data processing service."""
    try:
        # Import data processing service from main to avoid circular imports
        from .main import data_processing_service

        # Check each service's health
        data_processing_health = await data_processing_service.health_check()

        # Create response with boolean values
        return HealthResponse(
            status=(
                "healthy"
                if data_processing_health["status"] == "healthy"
                else "unhealthy"
            ),
            components={
                "data_processing": data_processing_health["status"] == "healthy"
            },
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Health check failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(exception)}"
        ) from exception


@router.post(
    "/preprocess", response_model=ProcessedData, tags=["Data Procesing Services"]
)
async def preprocess(
    symbol: str = Query(..., description="Stock symbol"),
    model_type: str = Query(..., description="Type of model (lstm, prophet, xgboost)"),
    phase: str = Query(
        ...,
        description="The current ML pipeline phase (training, prediction or evaluation)",
    ),
    raw_data: RawStockDataInput = Body(..., description="Raw data to preprocess"),
):
    """
    Preprocess the raw input data for a given model type and pipeline phase.

    This endpoint standardizes and transforms raw input data to the format expected
    by the specified machine learning model (`model_type`). The behavior may vary
    depending on the `model_type` (consult the response model schema reference).
    """
    try:
        # Import data processing service from main to avoid circular imports
        from .main import data_processing_service

        # Preprocess the raw data
        preprocessed_data = await data_processing_service.preprocess_data(
            symbol=symbol, model_type=model_type, phase=phase, raw_data=raw_data.data
        )

        return ProcessedData(data=preprocessed_data)

    except Exception as exception:
        api_logger.error(f"Preprocessing failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Preprocessing failed: {str(exception)}"
        ) from exception


@router.post("/postprocess", response_model=None, tags=["Data Procesing Services"])
async def postprocess(
    symbol: str = Query(..., description="Stock symbol"),
    model_type: str = Query(..., description="Type of model (lstm, prophet, xgboost)"),
    phase: str = Query(
        ...,
        description="The current ML pipeline phase (training, prediction or evaluation)",
    ),
    predictions: PredictionInput = Body(
        ..., description="Predictions (ouputs of ML/AI models to postprocess"
    ),
):
    """Postprocess the output of a given model"""
    try:
        # Import data processing service from main to avoid circular imports
        from .main import data_processing_service

        # Postprocess the predictions
        postprocessed_data = await data_processing_service.postprocess_data(
            symbol=symbol,
            model_type=model_type,
            phase=phase,
            prediction=predictions.data,
        )

        return ProcessedData(data=postprocessed_data)
    except Exception as exception:
        api_logger.error(f"Postprocessing failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Postprocessing failed: {str(exception)}"
        ) from exception


@router.get(
    "/promote-scaler",
    response_model=ScalerPromotionResponse,
    tags=["Data Procesing Services"],
)
def promote_scaler(
    symbol: str = Query(..., description="Stock symbol"),
    model_type: str = Query(..., description="Type of model (lstm, prophet, xgboost)"),
):
    """Promote a training scaler to production (prediction) for a specific model and symbol"""
    try:
        # Import data processing service from main to avoid circular imports
        from .main import data_processing_service

        promotion_results = data_processing_service.promote_scaler(
            symbol=symbol, model_type=model_type
        )

        return ScalerPromotionResponse(
            status=promotion_results["status"],
            symbol=symbol,
            model_type=model_type,
            promoted=promotion_results["promoted"],
            message=promotion_results["message"],
            timestamp=datetime.now().isoformat(),
        )

    except Exception as exception:
        api_logger.error(f"Scaler promotion failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Scaler promotion failed: {str(exception)}"
        ) from exception
