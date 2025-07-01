from abc import ABC, abstractmethod
from core.types import ProcessedData


class ConfidenceCalculatorStrategy(ABC):
    """Base strategy for calculating the confidence score"""

    @abstractmethod
    def calculate(
        self,
        y_pred,
        prediction_input: ProcessedData,
    ) -> list[float]:
        pass
