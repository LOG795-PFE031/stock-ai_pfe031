"""
Response Schemas used for the Data Procesing Service API
"""

from typing import Dict, List, Optional, Any, Union
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


# Scalers response schemas


class ScalerPromotionResponse(BaseModel):
    """API scaler promotion response."""

    status: str
    symbol: str
    model_type: str
    promoted: bool
    message: str
    timestamp: str


# Raw data to process input schemas
class RawStockDataInput(BaseModel):
    """Raw stock data input for the data processing."""

    data: List[Dict[str, Any]]


# Data processing response schemas


class SingleProcessedData(BaseModel):
    """Single Processed data response"""

    X: Optional[
        Union[
            List[List[List[float]]],  # LSTM
            List[Dict[str, Any]],  # Prophet
            List[Dict[str, float]],  # XGBoost
        ]
    ] = None
    y: Optional[List[float]] = None
    feature_index_map: Optional[Dict[str, int]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SplitProcessedData(BaseModel):
    """Used during the training phase"""

    train: SingleProcessedData
    test: SingleProcessedData


class ProcessedData(BaseModel):
    """API preprocessed data response."""

    data: Union[SingleProcessedData, SplitProcessedData]


# Prediction to postprocess input schema
class PredictionInput(BaseModel):
    """Predictions doned by the AI/ML models that need postprocessing."""

    data: Union[
        List[float],  # List of numbers
        List[List[float]],  # List of list of numbers
        List[Dict[str, Any]],  # JSON (pandas like predictions)
    ]
