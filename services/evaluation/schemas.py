from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    components: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    

class MetaInfo(BaseModel):
    """API metadata information."""

    message: str
    version: str
    documentation: str
    endpoints: List[str]


class ModelMetadata(BaseModel):
    """Model metadata information."""

    version: str
    created_at: str
    last_used: str
    performance_metrics: Dict[str, float]
    training_params: Dict[str, Any]
    
    
class ModelInfo(BaseModel):
    """Model information."""

    symbol: str
    model_type: str
    metadata: ModelMetadata


class EvaluateRequest(BaseModel):
    true_target: List[float]
    pred_target: List[float]


class EvaluateResponse(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float
    
    
class ReadyForDeploymentRequest(BaseModel):
    candidate_metrics: Dict[str, float]
    live_metrics: Dict[str, float]


class ReadyForDeploymentResponse(BaseModel):
    ready_for_deployment: bool

    