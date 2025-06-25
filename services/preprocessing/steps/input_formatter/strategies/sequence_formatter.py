from typing import Tuple
import numpy as np

from .base_strategy import InputFormatterStrategy
from core.config import config


class SequenceFormatter(InputFormatterStrategy):
    def format(self, data, phase):
        # Convert the data to a numpy array
        data_np = data.to_numpy()

        X, y = self._create_sequences(data_np)

        if phase == "training":
            return X, y
        elif phase == "predicttion":
            # Returns the last sequence (for next day prediction)
            return X[-1]
        else:
            raise ValueError(
                f"Invalid phase '{phase}'. Expected 'training' or 'prediction'."
            )

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data."""

        sequence_length = config.preprocessing.SEQUENCE_LENGTH

        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : (i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
