from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processing steps in preprocessing.
    """

    def __init__(self, symbol: str, logger):
        """
        Args:
            symbol (str): Stock symbol
            logger : Logger
        """
        self.symbol = symbol
        self.logger = logger

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data.

        Args:
            data (pd.DataFrame): Input stock data.

        Returns:
            pd.DataFrame: Processed stock data.
        """
        pass
