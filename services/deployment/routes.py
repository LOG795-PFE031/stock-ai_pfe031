"""
API routes for the Deployment Service.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
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
)

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
            "/predict/historical",
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
        return exists

    except Exception as e:
        api_logger.error(f"Get production model failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Get production model failed: {str(e)}"
        ) from e
        
        
@router.post(
    "/predict",
    response_model=dict[Any, int],
    tags=["Deployment Services"],
)
async def predict(request: PredictionRequest):
    """
    Run a prediction on the given input using the specified model.
    """
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service
        
        result = await deployment_service.predict(
            model_identifier=request.model_identifier,
            X=request.X,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {e}"
        )
        

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
    try:
        # Import services from main to avoid circular imports
        from .main import deployment_service
        
        confidences = await deployment_service.calculate_prediction_confidence(
            model_type=request.model_type,
            symbol=request.symbol,
            prediction_input=request.prediction_input,
            y_pred=request.y_pred,
        )
        return ConfidenceResponse(confidences=confidences or [])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate prediction confidence: {e}",
        )
        
        
# Prediction endpoints
# @router.get(
#     "/predict", 
#     response_model=PredictionResponse, 
#     tags=["Deployment Services"])
# async def get_next_day_prediction(
#     model_type: str = Query(..., description="Type of prediction model to use"),
#     symbol: str = Query(
#         ..., description="Ticker symbol of the stock (e.g., AAPL, MSFT)"
#     ),
# ):
#     """Get stock price prediction for the next day."""
#     try:
#         # Import services from main to avoid circular imports
#         from api.main import orchestation_service

#         # Validate symbol
#         if not validate_stock_symbol(symbol):
#             raise HTTPException(
#                 status_code=400, detail=f"Invalid stock symbol: {symbol}"
#             )

#         # Get prediction using the new method
#         prediction = await orchestation_service.run_prediction_pipeline(
#             model_type=model_type, symbol=symbol
#         )

#         if prediction.get("status") == "success":
#             return prediction

#         error_msg = prediction.get("error", "Unknown error")
#         api_logger.error(f"Prediction failed for {symbol}: {error_msg}")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {error_msg}")

#     except Exception as e:
#         api_logger.error(f"Prediction failed: {str(e)}")
#         raise HTTPException(
#             status_code=500, detail=f"Prediction failed: {str(e)}"
#         ) from e


# @router.get(
#     "/predict/historical",
#     response_model=PredictionsResponse,
#     tags=["Deployment Services"],
# )
# async def get_historical_predictions(
#     model_type: str = Query(..., description="Type of prediction model to use"),
#     symbol: str = Query(
#         ..., description="Ticker symbol of the stock (e.g., AAPL, MSFT)"
#     ),
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None,
# ):
#     """Get historical predictions for a symbol."""
#     try:
#         # Import services from main to avoid circular imports
#         from api.main import orchestation_service

#         # Validate symbol
#         if not validate_stock_symbol(symbol):
#             raise HTTPException(
#                 status_code=400, detail=f"Invalid stock symbol: {symbol}"
#             )

#         # Get date range
#         start, end = get_date_range(start_date, end_date)

#         # Get predictions
#         predictions = await orchestation_service.run_historical_prediction_pipeline(
#             model_type=model_type, symbol=symbol, start_date=start, end_date=end
#         )

#         return predictions
#     except Exception as e:
#         api_logger.error(f"Historical predictions failed: {str(e)}")
#         raise HTTPException(
#             status_code=500, detail=f"Historical predictions failed: {str(e)}"
#         ) from e


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
        

# @router.post(
#     "/models/{model_name}/promote",
#     response_model=PromoteModelResponse,
#     tags=["Deployment"],
# )
# async def promote_model(
#     model_name: str,
#     payload: PromoteModelRequest,
# ):
#     try:
#         # Import services from main to avoid circular imports
#         from .main import deployment_service
        
#         result = await deployment_service.promote_model(
#             run_id=payload.run_id,
#             prod_model_name=model_name,
#         )
        
#         return PromoteModelResponse(**result)
    
#     except Exception as e:
#         api_logger.error(f"Promote model failed: {str(e)}")
#         raise HTTPException(
#             status_code=500, detail=f"Promote model failed: {str(e)}"
#         ) from e
                

# @router.post(
#     "/models/{model_identifier}/log_metrics",
#     tags=["Deployment"],
# )
# async def log_metrics(
#     model_identifier: str,
#     metrics: dict[str, float],
# ):
#     """
#     Log evaluation metrics for the specified model.
#     """
#     try:
#         # Import services from main to avoid circular imports
#         from .main import deployment_service
        
#         success: bool = await deployment_service.log_metrics(
#             model_identifier=model_identifier,
#             metrics=metrics,
#         )
#         return success
    
#     except Exception as e:
#         api_logger.error(f"Failed to log metrics for {model_identifier}: {e}")
#         raise HTTPException(
#             status_code=500, detail=f"Failed to log metrics: {e}"
#         ) from e
        


