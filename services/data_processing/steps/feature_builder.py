from .abstract import BaseDataProcessor

import numpy as np
import pandas as pd


class FeatureBuilder(BaseDataProcessor):
    """
    Class that processes stock data to generate additional features, including:

    - Returns (percentage and log returns)
    - Technical indicators (Moving Averages, Volatility, RSI, MACD)
    - Temporal features (Day of week, Month, Quarter)
    """

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Add returns features
            data = self._calculate_and_add_returns(data)

            # Add techical indicators features
            data = self._calculate_and_add_technical_indicators(data)

            # Add temporal features
            data = self._add_temporal_features(data)

            return data
        except Exception as e:
            raise RuntimeError(f"Error building new features from stock data") from e

    def _calculate_and_add_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the returns and the log returns of the stock data (Close price)

        Args:
            data (pd.DataFrame): Dataframe containing the stock data

        Returns:
            pd.DataFrame: Dataframe containing the stock data with the retuns calculated
        """
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            data = data.copy()

            # Calculate Returns (percentage change)
            data["Returns"] = data["Close"].pct_change()

            # Calculate Log Returns
            data["Log_Returns"] = np.log(1 + data["Close"].pct_change())

            # Replace NaN values with forward fill then backward fill
            data = data.ffill().bfill()

            return data

        except Exception as e:
            raise RuntimeError(f"Error calculating log returns.") from e

    def _calculate_and_add_technical_indicators(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate technical indicators for stock data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with additional technical indicators
        """
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            data = data.copy()

            # Calculate Moving Averages
            data["MA_5"] = data["Close"].rolling(window=5).mean()
            data["MA_20"] = data["Close"].rolling(window=20).mean()

            # Calculate Volatility (20-day standard deviation of returns)
            data["Volatility"] = data["Returns"].rolling(window=20).std()

            # Calculate RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))

            # Calculate MACD
            exp1 = data["Close"].ewm(span=12, adjust=False).mean()
            exp2 = data["Close"].ewm(span=26, adjust=False).mean()
            data["MACD"] = exp1 - exp2
            data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

            # Ensure Adj Close exists (if not, use Close)
            if "Adj Close" not in data.columns:
                data["Adj Close"] = data["Close"]

            # Replace NaN values with forward fill then backward fill
            data = data.ffill().bfill()

            return data

        except Exception as e:
            raise RuntimeError(f"Error calculating technical indicators.") from e

    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features of the stock data. This features include :
        - Day of the week as a number
        - Month as a number
        - Quarter as a number (Q1, Q2, Q3, Q4)

        Args:
            data (pd.DataFrame): Dataframe containing the stock data

        Returns:
            pd.DataFrame: Dataframe containing the temporal features
        """
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            data = data.copy()

            # Add day of week column (the day week as a number)
            data["Day_of_week"] = data["Date"].dt.dayofweek

            # Add month column (month as a number)
            data["Month"] = data["Date"].dt.month

            # Add quarter column
            data["Quarter"] = data["Date"].dt.quarter

            return data

        except Exception as e:
            raise RuntimeError(f"Error adding the temporal features.") from e
