from typing import Tuple
import numpy as np

from .base_strategy import InputFormatterStrategy
from core.types import PreprocessedData
from core.config import config


class SequenceInputFormatter(InputFormatterStrategy):
    def format(self, data, phase):

        # Target index values
        target_index = data.columns.get_loc("Close")

        # Convert the data to a numpy array
        data_np = data.to_numpy()

        X, y = self._create_sequences(data_np, target_index)

        if phase == "training" or "evaluation":
            return PreprocessedData(X=X, y=y)
        elif phase == "prediction":
            # Returns the last sequence (for next day prediction)
            return PreprocessedData(X=X[-1])
        else:
            raise ValueError(
                f"Invalid phase '{phase}'. Expected 'training' or 'prediction'."
            )

    def _create_sequences(
        self, data: np.ndarray, target_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data."""

        sequence_length = config.preprocessing.SEQUENCE_LENGTH

        if sequence_length > len(data):
            raise ValueError(
                f"Sequence length ({sequence_length}) is greater than the length of the data ({len(data)})."
            )

        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : (i + sequence_length)])
            y.append(data[i + sequence_length, target_index])
        return np.array(X), np.array(y)
