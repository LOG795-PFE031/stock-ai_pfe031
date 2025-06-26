from .base_strategy import InputFormatterStrategy
from core.types import FormattedInput

from pandas.tseries.offsets import BDay
import pandas as pd


class ProphetFormatter(InputFormatterStrategy):
    def format(self, data, phase):
        if phase == "prediction":
            # Return a DataFrame with the next business day in a 'ds' column
            next_business_day = self._get_next_business_day()
            return FormattedInput(X=pd.DataFrame({"ds": [next_business_day]}))
        elif phase == "training":
            # Return the training data as-is (already in Prophet format)
            return FormattedInput(X=data)
        else:
            raise ValueError(
                f"Invalid phase '{phase}'. Expected 'training' or 'prediction'."
            )

    def _get_next_business_day(self) -> pd.Timestamp:
        return (pd.Timestamp.today().normalize() + BDay(1)).strftime("%Y-%m-%d")
