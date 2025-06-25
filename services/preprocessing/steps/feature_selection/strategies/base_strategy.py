from abc import ABC, abstractmethod
import pandas as pd


class FeatureSelectionStrategy(ABC):
    @abstractmethod
    def select(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
