import numpy as np
import pandas as pd

from core.types import ProcessedData
from ..abstract import BaseDataProcessor
from .output_strategies import (
    ProphetOutputFormatter,
    PandasSeriesFormatter,
    NumpyOutputFormatter,
    OutputFormatterStrategy,
)


class OutputFormatter(BaseDataProcessor):
    """
    Class that formats model output data for different
    model types using the appropriate strategy.
    """

    def __init__(self, model_type: str):
        self.model_type = model_type

    def process(self, data) -> np.ndarray:
        """
        Formats model output data

        Args:
            data (Any): Model output data

        Returns:
            np.ndarray: Formatted model output data
        """
        try:

            # Extract the targets
            y = data.y

            if isinstance(y, list) or isinstance(y, np.ndarray):
                y = np.array(y)  # Convert to numpy array (for list instance)
                data_formatter = NumpyOutputFormatter()
            elif isinstance(y, pd.Series):
                data_formatter = PandasSeriesFormatter()
            else:
                data_formatter = self._get_data_formatter()

            formatted_y = data_formatter.format(y)
            return ProcessedData(y=formatted_y)
        except Exception as e:
            raise RuntimeError("Error while formatting (output) the data") from e

    def _get_data_formatter(self) -> OutputFormatterStrategy:
        if self.model_type == "prophet":
            return ProphetOutputFormatter()
        else:
            raise NotImplementedError(
                f"No output formatter for model {self.model_type}"
            )
