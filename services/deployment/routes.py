"""
API routes for the Deployment Service.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import date, datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from core.config import config
from core.logging import logger
from core.types import ProcessedDataAPI
from core.utils import validate_stock_symbol, get_date_range
from .schemas import (
    HealthResponse,
    MetaInfo,
    ModelMlflowInfo,
    ModelListMlflowResponse,
    PredictionResponse,
    PredictionRequest,
    ConfidenceResponse,
    ConfidenceRequest,
    ModelExistResponse,
    PromoteModelRequest,
    PromoteModelResponse,
    LogMetricsRequest,
    LogMetricsResponse,
)

from core.types import ProcessedData

# Create router
router = APIRouter()

# Initialize API logger
api_logger = logger["deployment"]


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
        "message": "Welcome to Stock AI Deployment Service API",
        "version": config.api.API_VERSION,
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/metrics",
            "/models",
            "/models/{model_name}",
            "/models/{prod_model_name}/exists"
            "/predict",
            "/calculate_prediction_confidence",
            "/models/{prod_model_name}/promote",
            "/models/{model_identifier}/log_metrics",
            "/cleanup",
        ],
    }
    

# Health check endpoint
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of all services."""
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service

        # Check each service's health
        deployment_health = await deployment_service.health_check()

        # Create response with boolean values
        return HealthResponse(
            status=(
                "healthy" if deployment_health["status"] == "healthy" else "unhealthy"
            ),
            components={"deployment_service": deployment_health["status"] == "healthy"},
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Health check failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(exception)}"
        ) from exception
        

# Get all available models
@router.get(
    "/models", 
    response_model=ModelListMlflowResponse,
    tags=["Model Management"])
async def list_models():
    """List all available models."""
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service
        
        # models = await deployment_service.list_models()
        # return models
        models = await deployment_service.list_models()
        response = ModelListMlflowResponse(
            models=models,
            total_models=len(models),
            timestamp=datetime.now().isoformat(),
        )
        return response
    except Exception as exception:
        api_logger.error(
            f"Failed to retrieve the list of available models: {str(exception)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get the models: {str(exception)}"
        ) from exception


@router.get(
    "/models/{model_name}",
    response_model=ModelMlflowInfo,
    tags=["Deployment Services"],
)
async def get_model_metadata(model_name: str):
    """Get metadata for a specific model."""
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service
        
        metadata = await deployment_service.get_model_metadata(model_name)
        return metadata
    except Exception as exception:
        api_logger.error(f"Failed to get model metadata: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model metadata: {str(exception)}"
        ) from exception


@router.get(
    "/models/{prod_model_name}/exists",
    response_model=ModelExistResponse,
    tags=["Deployment services"],
)
async def production_model_exists(prod_model_name: str):
    """
    Check whether a production model with the given name exists.
    """
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service
        
        exists = await deployment_service.production_model_exists(
            prod_model_name=prod_model_name
        )
        return {"exists": exists, "model_name": prod_model_name}

    except Exception as e:
        api_logger.error(f"Get production model failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Get production model failed: {str(e)}"
        ) from e
        
        
@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Deployment Services"],
)
async def predict(request: PredictionRequest):
    """
    Run a prediction on the given input using the specified model.
    """
    raw = request.X
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        X_input = pd.DataFrame(raw)
    elif isinstance(raw, list):
        X_input = np.array(raw)
    else:
        X_input = raw
        
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service
        
        result = await deployment_service.predict(
            model_identifier=request.model_identifier,
            X=X_input,
        )
        
        # !!! Ensure predictions is always a List for Pydantic schema
        predictions = result.get("predictions")
        
        # Handle different prediction types and convert to List
        if predictions is None:
            predictions = []
        elif isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        elif isinstance(predictions, (pd.DataFrame, pd.Series)):
            predictions = predictions.values.tolist() if hasattr(predictions, 'values') else predictions.tolist()
        elif not isinstance(predictions, list):
            # Handle single values or other types
            predictions = [predictions] if np.isscalar(predictions) else list(predictions)
        
        # Ensure all elements are JSON serializable (int, float, or basic types)
        clean_predictions = []
        for pred in predictions:
            if isinstance(pred, (np.integer, np.int64, np.int32)):
                clean_predictions.append(int(pred))
            elif isinstance(pred, (np.floating, np.float64, np.float32)):
                clean_predictions.append(float(pred))
            elif isinstance(pred, (np.bool_, bool)):
                clean_predictions.append(bool(pred))
            else:
                clean_predictions.append(pred)
        
        return PredictionResponse(
            predictions=clean_predictions,
            model_version=result.get("model_version", "unknown")
        )
        
    except Exception as e:
        api_logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {e}"
        ) from e
        

@router.post(
    "/calculate_prediction_confidence",
    response_model=ConfidenceResponse,
    tags=["Metrics"],
)
async def calculate_prediction_confidence(
    request: ConfidenceRequest,
):
    """
    Calculate and log prediction confidence for a model/symbol,
    emit to Prometheus, and return the raw confidence scores.
    """
    # Import services from main to avoid circular imports
    from .main import deployment_service

    # rebuild the ProcessedData object from json
    pi = request.prediction_input
    try:
        processed_input = ProcessedData(
            X = np.array(pi["X"]),
            y = np.array(pi["y"]) if pi.get("y") is not None else None,
            feature_index_map = pi.get("feature_index_map"),
            start_date = date.fromisoformat(pi["start_date"]) if pi.get("start_date") else None,
            end_date   = date.fromisoformat(pi["end_date"])   if pi.get("end_date")   else None,
        )
    except Exception as e:
        raise HTTPException(400, f"Bad prediction_input payload: {e}")

    try:
        confidences = await deployment_service.calculate_prediction_confidence(
            model_type=request.model_type,
            symbol=request.symbol,
            prediction_input=processed_input,
            y_pred=np.array(request.y_pred),
        )
        return ConfidenceResponse(confidences=confidences or [])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate prediction confidence: {e}",
        )
        
        
@router.post(
    "/models/{prod_model_name}/promote",
    response_model=PromoteModelResponse,
    tags=["Deployment Services"],
)
async def promote_model(
    prod_model_name: str,
    payload: PromoteModelRequest,
):
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service
        
        result = await deployment_service.promote_model(
            run_id=payload.run_id,
            prod_model_name=prod_model_name,
        )
        
        return PromoteModelResponse(**result)
    
    except Exception as e:
        api_logger.error(f"Promote model failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Promote model failed: {str(e)}"
        ) from e


@router.post(
    "/models/{model_identifier}/log_metrics",
    response_model=LogMetricsResponse,
    tags=["Deployment Services"],
)
async def log_metrics(
    model_identifier: str,
    payload: LogMetricsRequest,
):
    """
    Log evaluation metrics for the specified model.
    """
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service
        
        success = await deployment_service.log_metrics(
            model_identifier=model_identifier,
            metrics=payload.metrics,
        )
        return {"logged": success, "model_identifier": model_identifier}
    
    except Exception as e:
        api_logger.error(f"Failed to log metrics for {model_identifier}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to log metrics: {e}"
        ) from e


@router.post("/cleanup", response_model=Dict[str, Any], tags=["Deployment Services"])
async def cleanup_deployment_service():
    """Clean up the deployment service."""
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service

        # Clean up deployment service
        result = await deployment_service.cleanup()

        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Deployment service cleanup failed: {result.get('message', 'Unknown error')}",
            )

        return result

    except Exception as exception:
        api_logger.error(f"Deployment service cleanup failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Deployment service cleanup failed: {str(exception)}"
        ) from exception
        
                

        


