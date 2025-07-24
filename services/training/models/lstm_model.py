from .base_model import BaseModel
from .saving_strategies import KerasSaver
from ..trainers import LSTMTrainer
from ..predictors import LSTMPredictor
from ..models.base_model import BaseModel
from ..model_registry import ModelRegistry

# Constant for the trainer name
MODEL_NAME = "lstm"


@ModelRegistry.register(MODEL_NAME)
class LSTMModel(BaseModel):
    def __init__(
        self,
        symbol,
        saver=KerasSaver(),
        trainer=LSTMTrainer(),
        predictor=LSTMPredictor(),
    ):
        super().__init__(MODEL_NAME, symbol, saver, trainer, predictor)
