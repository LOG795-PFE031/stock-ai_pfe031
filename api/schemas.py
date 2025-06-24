"""
Pydantic models for API request/response validation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ModelType(str, Enum):
    """Available model types."""

    LSTM = "lstm"
    PROPHET = "prophet"


class MetaInfo(BaseModel):
    """API metadata information."""

    message: str
    version: str
    documentation: str
    endpoints: List[str]


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


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    symbol: str
    date: str
    predicted_price: float
    confidence: float
    model_type: ModelType
    timestamp: str


class PredictionsResponse(BaseModel):
    """Historical predictions response schema."""

    symbol: str
    predictions: List[PredictionResponse]
    timestamp: str


class NewsAnalysisResponse(BaseModel):
    """News analysis response schema."""

    symbol: str
    period: Dict[str, str]
    total_articles: int
    sentiment_metrics: Dict[str, float]
    articles: List[Dict[str, Any]]
    model_version: str


class TrainingTrainersResponse(BaseModel):
    """Trainers getter response schema."""

    status: str
    timestamp: str
    result: Dict[str, List[str]] = None
    error: Optional[str] = None


class TrainingResponse(BaseModel):
    """Model training response."""

    symbol: str
    model_type: str
    model_version: str
    training_history: Dict[str, List[float]]
    metrics: Dict[str, Any]
    timestamp: str


class TrainingStatusResponse(BaseModel):
    """Training status response."""

    status: str
    symbol: str
    model_type: str
    timestamp: str
    result: Optional[TrainingResponse] = None
    error: Optional[str] = None


class TrainingTask(BaseModel):
    """Individual training task information."""

    symbol: str
    model_type: str
    status: str
    timestamp: str


class TrainingTasksResponse(BaseModel):
    """List of active training tasks."""

    tasks: List[TrainingTask]
    total_tasks: int
    timestamp: str


class DataUpdateResponse(BaseModel):
    """Data update response."""

    symbol: str
    stock_data_updated: bool
    timestamp: str
    stock_records: int
    news_articles: int


class StockDataResponse(BaseModel):
    """Stock data response."""

    symbol: str
    name: str
    data: List[Dict[str, Any]]
    meta: MetaInfo
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class NewsDataResponse(BaseModel):
    """News data response."""

    symbol: str
    articles: List[Dict[str, Any]]
    total_articles: int
    sentiment_metrics: Dict[str, float]
    meta: MetaInfo


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


class ModelMetadataResponse(BaseModel):
    """Detailed model metadata response."""

    symbol: str
    model_type: str
    version: str
    metadata: ModelMetadata
    timestamp: str


class DirectDisplayResponse(BaseModel):
    """Direct display response schema."""

    symbol: str
    predictions: List[PredictionResponse]
    next_day: PredictionResponse
    news_analysis: NewsAnalysisResponse
    timestamp: str
