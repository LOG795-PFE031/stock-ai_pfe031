from services.data_processing.abstract import BaseDataProcessor
from core.types import PreprocessedData

from .input_strategies import (
    InputFormatterStrategy,
    SequenceInputFormatter,
    ProphetInputFormatter,
)


class InputFormatter(BaseDataProcessor):
    """
    Class that formats the data for different model types using
    appropriate formatting strategies.
    """

    def __init__(self, model_type: str, phase: str):
        self.model_type = model_type
        self.phase = phase

    def process(self, data) -> PreprocessedData:
        """
        Formats the data for the different model types input.

        Args:
            data (pd.DataFrame): Input stock data.

        Returns:
            FormattedData: Either X and Y (for training phase) or just X (for prediction phase)
        """
        try:
            data_formatter = self._get_data_formatter()
            formatted_data = data_formatter.format(data, self.phase)
            return formatted_data
        except Exception as e:
            raise RuntimeError("Error while formatting (input) the data") from e

    def _get_data_formatter(self) -> InputFormatterStrategy:
        if self.model_type == "lstm":
            return SequenceInputFormatter()
        elif self.model_type == "prophet":
            return ProphetInputFormatter()
        else:
            raise NotImplemented(f"No input formatter for model {self.model_type}")
