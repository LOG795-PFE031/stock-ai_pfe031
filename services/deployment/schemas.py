from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

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


class ModelVersionInfo(BaseModel):
    version: str
    stage: Optional[str]
    status: Optional[str]
    run_id: Optional[str]
    creation_timestamp: Optional[int]
    last_updated_timestamp: Optional[int]


class ModelMetadata(BaseModel):
    """Model metadata information."""

    version: str
    created_at: str
    last_used: str
    performance_metrics: Dict[str, float]
    training_params: Dict[str, Any]
    

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
    
    
class PromoteModelRequest(BaseModel):
    run_id: str


class PromoteModelResponse(BaseModel):
    deployed: bool
    model_name: str
    version: int
    run_id: str
    

class PredictionConfidenceRequest(BaseModel):
    # model_type: str
    # symbol: str
    # prediction_input: Any
    # y_pred: Any
    model_type: str
    symbol: str
    X: List[Any]    
    feature_index_map: Dict[str, int]
    y_pred: List[Any]
    
# The one in deployment_service.py predict()
class PredictionRequest(BaseModel):
    model_identifier: str
    X: Any