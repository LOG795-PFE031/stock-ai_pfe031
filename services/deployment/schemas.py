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
    
    
class ModelListResponse(BaseModel):
    """List of available models."""

    models: List[ModelInfo]
    total_models: int
    timestamp: str
    

class ModelVersionInfo(BaseModel):
    version: str
    stage: Optional[str]
    status: Optional[str]
    run_id: Optional[str]
    creation_timestamp: Optional[int]
    last_updated_timestamp: Optional[int]


class ModelMlflowInfo(BaseModel):
    name: str
    description: Optional[str]
    creation_timestamp: Optional[int]
    last_updated_timestamp: Optional[int]
    tags: Dict[str, str]
    aliases: Dict[str, Any]
    latest_versions: List[ModelVersionInfo]
    

class ModelListMlflowResponse(BaseModel):
    """List of available models in MLflow."""

    models: List[ModelMlflowInfo]
    total_models: int
    timestamp: str


class ModelMetadataResponse(BaseModel):
    """Detailed model metadata response."""

    symbol: str
    model_type: str
    version: str
    metadata: ModelMetadata
    timestamp: str
    

class PredictionRequest(BaseModel):
    model_identifier: str
    X: List[Union[Dict[str, Any], List[Any]]]


class PredictionResponse(BaseModel):
    predictions: List[Union[int, float, Any]]
    model_version: str


class ModelExistResponse(BaseModel):
    exists: bool
    model_name: str


class PredictionsResponse(BaseModel):
    """Historical predictions response schema."""

    symbol: str
    predictions: List[PredictionResponse]
    timestamp: str
    

class ConfidenceRequest(BaseModel):
    model_type: str
    symbol: str
    prediction_input: Dict[str, Any]
    y_pred: List[Any]

class ConfidenceResponse(BaseModel):
    confidences: List[float]


class PromoteModelRequest(BaseModel):
    run_id: str


class PromoteModelResponse(BaseModel):
    deployed: bool
    model_name: str
    version: Union[int, str]
    run_id: str


class LogMetricsRequest(BaseModel):
    metrics: Dict[str, Any]
    
    
class LogMetricsResponse(BaseModel):
    logged: bool
    model_identifier: str

