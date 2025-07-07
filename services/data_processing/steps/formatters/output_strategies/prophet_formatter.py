from .base_strategy import OutputFormatterStrategy
import pandas as pd


class ProphetOutputFormatter(OutputFormatterStrategy):
    def format(self, targets: pd.DataFrame):
        return targets["yhat"].values
