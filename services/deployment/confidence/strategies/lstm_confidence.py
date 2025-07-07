from .base_strategy import ConfidenceCalculatorStrategy

import numpy as np


class LSTMConfidenceCalculator(ConfidenceCalculatorStrategy):
    def calculate(self, y_pred, prediction_input):

        try:
            # Get the "Close" index
            close_index = prediction_input.feature_index_map["Close"]

            # Get the sequence from the prediction input
            sequences = prediction_input.X

            confidences = []

            # Loop through each sequence
            for i, sequence in enumerate(sequences):
                close_series = sequence[:, close_index]

                # Calculate historical volatility for this sequence
                historical_volatility = np.std(close_series) / np.mean(close_series)

                # Prediction magnitude relative to last close price in the sequence
                last_price = close_series[-1]
                predicted_value = y_pred[i, 0]

                prediction_magnitude = (
                    abs(predicted_value - last_price) / abs(last_price)
                    if last_price != 0
                    else float("inf")
                )

                # Confidence logic
                volatility_factor = np.exp(-2 * historical_volatility)
                magnitude_factor = np.exp(-3 * prediction_magnitude)

                confidence = 0.5 * volatility_factor + 0.5 * magnitude_factor
                confidence = np.clip(confidence, 0, 1)

                confidences.append(round(confidence, 3))

            return confidences
        except Exception as e:
            raise e
