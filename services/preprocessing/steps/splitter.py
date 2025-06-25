from typing import Tuple
import pandas as pd

from ..abstract import BaseDataProcessor
from core.config import config


class DataSplitter(BaseDataProcessor):

    def __init__(self, symbol, logger):
        super().__init__(symbol, logger)

    def process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into a train and test datasets

        Args:
            data (pd.DataFrame): Input stock data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
        """

        # Get the datasets sizes
        total_size = len(data)
        train_size = int(total_size * config.preprocessing.TRAINING_SPLIT_RATIO)
        test_size = total_size - train_size

        # Split into train and test datasets
        training_data = data.head(train_size)
        test_data = data.tail(test_size)

        return training_data, test_data
