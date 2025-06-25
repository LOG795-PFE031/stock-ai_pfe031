from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
import numpy as np


@dataclass
class FormattedInput:
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    y: Optional[Union[pd.Series, np.ndarray]] = None
