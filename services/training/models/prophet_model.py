from .base_model import BaseModel
from .saving_strategies import JoblibSaver
from ..trainers import ProphetTrainer
from ..predictors import ProphetPredictor
from ..models.base_model import BaseModel
from ..model_registry import ModelRegistry

# Constant for the trainer name
MODEL_NAME = "prophet"


@ModelRegistry.register(MODEL_NAME)
class ProphetModel(BaseModel):
    def __init__(
        self,
        symbol,
        saver=JoblibSaver(),
        trainer=ProphetTrainer(),
        predictor=ProphetPredictor(),
    ):
        super().__init__(MODEL_NAME, symbol, saver, trainer, predictor)
