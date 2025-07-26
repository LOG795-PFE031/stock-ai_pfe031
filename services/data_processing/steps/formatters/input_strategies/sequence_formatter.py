from typing import Tuple
import numpy as np

from .base_strategy import InputFormatterStrategy
from ..types import ProcessedData
from core.config import config


class SequenceInputFormatter(InputFormatterStrategy):

    def __init__(self):
        super().__init__()
        self.sequence_length = config.preprocessing.SEQUENCE_LENGTH

    def format(self, data, phase):

        # Target index values
        target_index = data.columns.get_loc("Close")

        # Convert the data to a numpy array
        data_np = data.to_numpy()

        if phase == "training" or phase == "evaluation":
            X, y = self._create_sequences(data_np, target_index)
            return ProcessedData(X=X, y=y)
        elif phase == "prediction":
            # Returns the last sequence (for next day prediction)
            return ProcessedData(X=self._get_last_sequence(data))
        else:
            raise ValueError(
                f"Invalid phase '{phase}'. Expected 'training' or 'prediction'."
            )

    def _create_sequences(
        self, data: np.ndarray, target_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data."""

        if self.sequence_length > len(data):
            raise ValueError(
                f"Sequence length ({self.sequence_length}) is greater than the length of the data ({len(data)})."
            )

        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : (i + self.sequence_length)])
            y.append(data[i + self.sequence_length, target_index])
        return np.array(X), np.array(y)

    def _get_last_sequence(self, data: np.ndarray) -> np.ndarray:
        """Get the last sequence (for prediction)"""
        sequence = data[-self.sequence_length :]
        # To match model input (1, sequence_length, num_features)
        return np.expand_dims(sequence, axis=0)
