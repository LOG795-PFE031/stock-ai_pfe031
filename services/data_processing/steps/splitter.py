from typing import Tuple
from sklearn.model_selection import train_test_split

from .abstract import BaseDataProcessor
from core.config import config
from core.types import ProcessedData


class DataSplitter(BaseDataProcessor):

    def process(self, data: ProcessedData) -> Tuple[ProcessedData, ProcessedData]:
        """
        Split the preprocessed data into training and test datasets.

        Args:
            data (PreprocessedData): Preprocessed features and targets.

        Returns:
            Tuple[PreprocessedData, PreprocessedData]: Train and test datasets.
        """

        try:
            # Test ratio
            test_ratio = 1 - config.preprocessing.TRAINING_SPLIT_RATIO

            # Get the datasets sizes
            X_train, X_test, y_train, y_test = train_test_split(
                data.X, data.y, test_size=test_ratio, shuffle=False
            )

            return (
                ProcessedData(
                    X=X_train, y=y_train, feature_index_map=data.feature_index_map
                ),
                ProcessedData(
                    X=X_test, y=y_test, feature_index_map=data.feature_index_map
                ),
            )
        except Exception as e:
            raise RuntimeError("Error while splitting the data") from e
