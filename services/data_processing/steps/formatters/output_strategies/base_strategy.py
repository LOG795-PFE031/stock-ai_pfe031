from abc import ABC, abstractmethod
import numpy as np


class OutputFormatterStrategy(ABC):
    @abstractmethod
    def format(self, targets) -> np.ndarray:
        pass
