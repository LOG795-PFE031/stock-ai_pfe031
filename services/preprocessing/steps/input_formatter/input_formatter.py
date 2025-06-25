from services.preprocessing.abstract import BaseDataProcessor
from services.preprocessing.types import FormattedInput
from .strategies import InputFormatterStrategy, SequenceFormatter, ProphetFormatter


class InputFormatter(BaseDataProcessor):
    """
    Class that formats input data for different model types using
    appropriate formatting strategies.
    """

    def __init__(self, symbol, logger, model_type: str, phase: str):
        super().__init__(symbol, logger)
        self.model_type = model_type
        self.phase = phase

    def process(self, data) -> FormattedInput:
        """
        Formats the data for the different model types input.

        Args:
            data (pd.DataFrame): Input stock data.

        Returns:
            FormattedData: Either X and Y (for training phase) or just X (for prediction phase)
        """
        input_formatter = self._get_input_formatter()

        return input_formatter.format(data, self.phase)

    def _get_input_formatter(self) -> InputFormatterStrategy:
        if self.model_type == "lstm":
            return SequenceFormatter()
        elif self.model_type == "prophet":
            return ProphetFormatter()
        else:
            raise NotImplemented(f"No input formatter for model {self.model_type}")
