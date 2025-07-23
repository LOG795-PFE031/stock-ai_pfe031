import pandas as pd
import numpy as np

from .base_strategy import ConfidenceCalculatorStrategy


class XGBoostConfidenceCalculator(ConfidenceCalculatorStrategy):
    """Class responsible to calculate the confidence score for the XGBoost model"""

    def calculate(self, y_pred: pd.DataFrame, prediction_input):
        try:
            confidences = []

            for i in range(len(prediction_input)):

                # Retrieve the prediction input and predicted price
                x_input = prediction_input.iloc[[i]]
                predicted_price = y_pred[i]

                # Retrieve high, low and close prices and volume from the prediciton input
                high, low, close, volume = (
                    x_input["High"].iloc[0],
                    x_input["Low"].iloc[0],
                    x_input["Close"].iloc[0],
                    x_input["Volume"].iloc[0],
                )

                # Price range tightness (normalized to closing price)
                price_range = high - low
                range_ratio = price_range / close

                # Volume ratio (normalized)
                volume_ratio = np.log1p(volume) / np.log1p(1e6)

                # Close proximity to midpoint (0=perfect midpoint)
                midpoint = (high + low) / 2
                close_balance = 1 - abs(close - midpoint) / (high - low)

                # Percentage change from the last close price
                predicted_pct_change = abs((predicted_price - close) / close)

                # Scales the predicted percentage change relative to the typical change
                typical_change = 0.01  # This value is
                prediction_confidence = 1 - np.tanh(
                    predicted_pct_change / typical_change
                )

                # Combine the different factors to have the confidence
                confidence = (
                    0.3 * (1 - np.tanh(range_ratio * 5))
                    + 0.2 * np.clip(volume_ratio, 0, 1)
                    + 0.2 * close_balance
                    + 0.3 * prediction_confidence
                )

                # Keep the confidence between the range [0.1, 0.9]
                confidence = np.clip(confidence, 0.1, 0.9)
                confidences.append(confidence)

            return confidences
        except Exception as e:
            raise e
