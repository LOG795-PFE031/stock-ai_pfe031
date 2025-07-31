from dataclasses import dataclass


# Represents the evaluation metrics expected from a model
@dataclass
class Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float
