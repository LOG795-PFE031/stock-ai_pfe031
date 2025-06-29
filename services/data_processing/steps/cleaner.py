from ..abstract import BaseDataProcessor

import pandas as pd


class DataCleaner(BaseDataProcessor):
    """
    Cleans stock data by resetting index, parsing dates, and filling missing values.
    """

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data (pd.DataFrame).

        Args:
            data (pd.DataFrame): Raw stock data.

        Returns:
            pd.DataFrame: Cleaned data.

        """
        try:
            data["Date"] = pd.to_datetime(data["Date"], format="mixed", utc=True)
            data = data.ffill().bfill()

            return data
        except Exception as e:
            raise RuntimeError(f"Error cleaning stock data") from e
