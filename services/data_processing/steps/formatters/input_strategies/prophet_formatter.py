from .base_strategy import InputFormatterStrategy
from ..types import ProcessedData

from pandas.tseries.offsets import BDay
import pandas as pd


class ProphetInputFormatter(InputFormatterStrategy):
    def format(self, data, phase):
        if phase == "prediction":

            # Get the latest data
            latest_data = data.tail(1).reset_index(drop=True)

            # Get the next available business date and update it (in the ds column)
            latest_data.at[0, "ds"] = self._get_next_business_day()

            return ProcessedData(X=latest_data)

        elif phase == "training" or "evaluation":
            # Return the training data as-is (already in Prophet format)
            y = data["y"].values
            return ProcessedData(X=data, y=y)
        else:
            raise ValueError(
                f"Invalid phase '{phase}'. Expected 'training' or 'prediction'."
            )

    def _get_next_business_day(self) -> pd.Timestamp:
        return (pd.Timestamp.today().normalize() + BDay(1)).strftime("%Y-%m-%d")
