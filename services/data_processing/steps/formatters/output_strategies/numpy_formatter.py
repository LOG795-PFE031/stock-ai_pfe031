import numpy as np

from .base_strategy import OutputFormatterStrategy


class NumpyOutputFormatter(OutputFormatterStrategy):
    def format(self, targets):
        targets = np.array(targets)
        if targets.ndim > 1:
            # Flatten to 1 dimension
            targets = targets.ravel()
        return targets
