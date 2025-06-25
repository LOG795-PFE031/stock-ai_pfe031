from abc import ABC, abstractmethod
import pandas as pd


class InputFormatterStrategy(ABC):
    @abstractmethod
    def format(self, data: pd.DataFrame, phase: str):
        pass
