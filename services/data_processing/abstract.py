from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processing steps in preprocessing.
    """

    @abstractmethod
    def process(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Process the data.

        Args:
            data (pd.DataFrame): Input stock data.

        Returns:
            pd.DataFrame|np.ndarray: Processed stock data.
        """
        pass
