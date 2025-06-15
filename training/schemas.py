from pydantic import BaseModel


# Represents the evaluation metrics expected from a model
class Metrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float
