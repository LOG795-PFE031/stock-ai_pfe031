from services.training.models.base_model import BaseModel
from services.training.model_registry import ModelRegistry
from services.training.trainers import XGBoostTrainer
from services.training.predictors import XGBoostPredictor
from .base_model import BaseModel
from .saving_strategies import JoblibSaver

# Constant for the trainer name
MODEL_NAME = "xgboost"


@ModelRegistry.register(MODEL_NAME)
class XGBoostModel(BaseModel):
    def __init__(
        self,
        symbol,
        saver=JoblibSaver(),
        trainer=XGBoostTrainer(),
        predictor=XGBoostPredictor(),
    ):
        super().__init__(MODEL_NAME, symbol, saver, trainer, predictor)
