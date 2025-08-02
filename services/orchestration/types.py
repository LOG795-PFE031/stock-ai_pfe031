from dataclasses import dataclass
from typing import Generic, Optional, TypeVar
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
