from .strategies import (
    ConfidenceCalculatorStrategy,
    LSTMConfidenceCalculator,
    ProphetConfidenceCalculator,
)


class ConfidenceCalculator:
    def __init__(self, model_type: str):
        self.model_type = model_type

    def calculate_confidence(self, y_pred, prediction_input):
        try:
            confidence_calculator = self._get_confidence_calculator()
            confidence = confidence_calculator.calculate(y_pred, prediction_input)
            return confidence
        except Exception as e:
            raise RuntimeError(
                f"Error while calculating the confidence score for {self.model_type} model"
            ) from e

    def _get_confidence_calculator(self) -> ConfidenceCalculatorStrategy:
        """
        Returns the appropriate confidence calculator strategy based on the model type.
        """

        if self.model_type == "lstm":
            return LSTMConfidenceCalculator()
        elif self.model_type == "prophet":
            return ProphetConfidenceCalculator()
        else:
            # There is no confidence calculator for other models
            return None
