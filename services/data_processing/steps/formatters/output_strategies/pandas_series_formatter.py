import pandas as pd

from .base_strategy import OutputFormatterStrategy


class PandasSeriesFormatter(OutputFormatterStrategy):
    def format(self, targets: pd.Series):
        return targets.values
