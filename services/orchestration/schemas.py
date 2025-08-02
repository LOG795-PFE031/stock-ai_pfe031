"""
Response Schemas used for the Orchestration Service API
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

# System response schemas


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    components: Dict[str, bool]


class MetaInfo(BaseModel):
    """API metadata information."""

    message: str
    version: str
    documentation: str
    endpoints: List[str]


# Orchestration response schemas


class TrainingResponse(BaseModel):
    """Model training response."""

    status: str
    symbol: str
    model_type: str
    training_results: Dict[str, Any]
    metrics: Dict[str, float]
    deployment_results: Optional[Dict[str, Any]] = None
    timestamp: str


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    status: str
    symbol: str
    date: str
    predicted_price: float
    confidence: float
    model_type: str
    model_version: int
    timestamp: str


class PredictionsResponse(BaseModel):
    """Historical predictions response schema."""

    symbol: str
    predictions: List[PredictionResponse]
    timestamp: str


class EvaluateResponse(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float
