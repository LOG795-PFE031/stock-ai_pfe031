from .base_strategy import ConfidenceCalculatorStrategy

import pandas as pd
import numpy as np


class ProphetConfidenceCalculator(ConfidenceCalculatorStrategy):
    def calculate(self, y_pred: pd.DataFrame, _):
        try:
            confidences = []
            for index, prediction in y_pred.iterrows():

                # Calculate confidence based on prediction interval width
                interval_width = prediction["yhat_upper"] - prediction["yhat_lower"]
                relative_width = interval_width / abs(prediction["yhat"])

                # Convert to confidence score (0-1)
                # Use a more conservative sigmoid function that accounts for stock market uncertainty
                # The parameters are tuned to give reasonable confidence scores for typical stock price movements
                raw_confidence = 1 / (1 + np.exp(10 * relative_width - 1))

                # Round to 3 decimal places for cleaner output
                confidence = round(raw_confidence, 3)

                confidences.append(confidence)

            return confidences
        except Exception as e:
            raise e
