from abc import ABC, abstractmethod
import pandas as pd

from core.types import ProcessedData


class InputFormatterStrategy(ABC):
    @abstractmethod
    def format(self, data: pd.DataFrame, phase: str) -> ProcessedData:
        pass
