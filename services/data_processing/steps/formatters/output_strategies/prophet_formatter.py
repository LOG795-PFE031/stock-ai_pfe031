from .base_strategy import OutputFormatterStrategy
import pandas as pd


class ProphetOutputFormatter(OutputFormatterStrategy):
    def format(self, targets: pd.DataFrame):
        try:
            formatted_targets = targets["yhat"].values
            return formatted_targets
        except Exception as exception:
            raise RuntimeError(
                f"Error formatting output for ¨Prophet: {str(exception)}"
            ) from exception
