from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, List, Dict, Union, Any
from pydantic import BaseModel
from datetime import datetime
import numpy as np

TX = TypeVar("TX")  # Type of X (Usually a numpy array or a pandas DataFrame)


# Represents the processed output from the processing service.
# TODO Move this type inside services/data-processing/
@dataclass
class ProcessedData(Generic[TX]):
    X: Optional[TX] = None
    y: Optional[np.ndarray] = None
    feature_index_map: Optional[dict[str, int]] = None
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


# Model for Prophet (expects DataFrame input in JSON form)
class ProphetInput(BaseModel):
    X: Optional[List[Dict[str, Any]]]
    y: Optional[List[float]] = None
    feature_index_map: Optional[Dict[str, int]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# Model for LSTM (expects sequence-like input)
class LSTMInput(BaseModel):
    X: Optional[List[List[List[float]]]]
    y: Optional[List[float]] = None
    feature_index_map: Optional[Dict[str, int]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# Model for Prophet (expects DataFrame input in JSON form)
class XGBoostInput(BaseModel):
    X: Optional[List[Dict[str, float]]]
    y: Optional[List[float]] = None
    feature_index_map: Optional[Dict[str, int]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# TODO When the class ProcessedData will be deleted, we can use this name (instead of ProcessedDataAPI)
class ProcessedDataAPI(BaseModel):
    data: Union[LSTMInput, ProphetInput, XGBoostInput]


# Represents the evaluation metrics expected from a model
@dataclass
class Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float
