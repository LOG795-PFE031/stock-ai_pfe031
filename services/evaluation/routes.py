"""
API routes for the Deployment Service.
"""

from typing import Dict, Any, List, Optional

from datetime import date, datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from core.config import config
from core.logging import logger

from .schemas import (
    HealthResponse,
    MetaInfo,
    EvaluateRequest,
    EvaluateResponse,
    ReadyForDeploymentRequest,
    ReadyForDeploymentResponse,
)

# Create router
router = APIRouter()

# Initialize API logger
api_logger = logger["evaluation"]


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
        "message": "Welcome to Stock AI Evaluation Service API",
        "version": config.api.API_VERSION,
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/metrics",
            "/cleanup",
        ],
    }
    

# Health check endpoint
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of all services."""
    try:
        # Import services from main to avoid circular imports
        from .main import evaluation_service

        # Check each service's health
        evaluation_health = await evaluation_service.health_check()

        # Create response with boolean values
        return HealthResponse(
            status=(
                "healthy" if evaluation_health["status"] == "healthy" else "unhealthy"
            ),
            components={"evaluation_service": evaluation_health["status"] == "healthy"},
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as exception:
        api_logger.error(f"Health check failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(exception)}"
        ) from exception
        

@router.post(
    "/models/{model_type}/{symbol}/evaluate",
    response_model=EvaluateResponse,
    tags=["Evaluation Services"],
)
async def evaluate_model(
    model_type: str,
    symbol: str,
    payload: EvaluateRequest,
) -> EvaluateResponse:
    """
    Evaluate predictions against ground truth for a given model_type and symbol.
    """
    try:
        # Import services from main to avoid circular imports
        from .main import evaluation_service
        
        metrics: Dict[str, float] = await evaluation_service.evaluate(
            y_true=payload.true_target,
            y_pred=payload.pred_target,
            model_type=model_type,
            symbol=symbol,
        )
        return EvaluateResponse(**metrics)
    except Exception as e:
        api_logger.error(f"Evaluation failed for {model_type}/{symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}") from e


@router.post(
    "/ready_for_deployment",
    response_model=ReadyForDeploymentResponse,
    tags=["Evaluation Services"],
)
async def check_ready_for_deployment(
    payload: ReadyForDeploymentRequest,
) -> ReadyForDeploymentResponse:
    """
    Compare candidate vs. live metrics and decide if the candidate should be deployed.
    """
    try:
        # Import services from main to avoid circular imports
        from .main import evaluation_service
        
        result: bool = await evaluation_service.is_ready_for_deployment(
            candidate_metrics=payload.candidate_metrics,
            live_metrics=payload.live_metrics,
        )
        return ReadyForDeploymentResponse(ready_for_deployment=result)
    except Exception as e:
        api_logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Readiness check failed: {e}") from e
    

@router.post("/cleanup", response_model=Dict[str, Any], tags=["Evaluation Services"])
async def cleanup_evaluation_service():
    """Clean up the evaluation service."""
    try:
        # Import services from main to avoid circular imports
        from .main import evaluation_service

        # Clean up evaluation service
        result = await evaluation_service.cleanup()

        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation service cleanup failed: {result.get('message', 'Unknown error')}",
            )

        return result

    except Exception as exception:
        api_logger.error(f"Evaluation service cleanup failed: {str(exception)}")
        raise HTTPException(
            status_code=500, detail=f"Evaluation service cleanup failed: {str(exception)}"
        ) from exception
        