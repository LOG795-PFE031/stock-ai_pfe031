from dataclasses import dataclass
from typing import Generic, Optional, TypeVar
import pandas as pd
import numpy as np

TX = TypeVar("TX")  # Type of X
TY = TypeVar("TY")  # Type of y


# Represents the processed output from the preprocessing service.
# This is also used as the input to the training service for model training.
@dataclass
class FormattedInput(Generic[TX, TY]):
    X: Optional[TX] = None
    y: Optional[TY] = None


# Represents the evaluation metrics expected from a model
@dataclass
class Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float
