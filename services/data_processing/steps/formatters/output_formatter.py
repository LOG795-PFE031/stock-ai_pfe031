import numpy as np

from .types import ProcessedData
from ..abstract import BaseDataProcessor
from .output_strategies import (
    ProphetOutputFormatter,
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

    def process(self, data: ProcessedData) -> ProcessedData:
        """
        Formats model output data

        Args:
            data (ProcessedData): Model output data

        Returns:
            ProcessedData: Formatted model output data
        """
        try:

            # Extract the targets
            y = data.y

            if isinstance(y, np.ndarray):
                data_formatter = NumpyOutputFormatter()
            else:
                data_formatter = self._get_data_formatter()

            # Format the targets
            formatted_y = data_formatter.format(y)
            return ProcessedData(y=formatted_y)
        except Exception as e:
            raise RuntimeError(
                f"Error while formatting (output) the data : {str(e)}"
            ) from e

    def _get_data_formatter(self) -> OutputFormatterStrategy:
        if self.model_type == "prophet":
            return ProphetOutputFormatter()
        else:
            raise NotImplementedError(
                f"No output formatter for model {self.model_type}"
            )
