from core.logging import logger
from services import BaseService

from .flows import (
    run_evaluation_pipeline,
    run_prediction_pipeline,
    run_training_pipeline,
)
from services import BaseService
from services import PredictionService
from services import TrainingService
from services import PreprocessingService
from services import DataService
from core.logging import logger


class OrchestrationService(BaseService):

    def __init__(
        self,
        data_service: DataService,
        preprocessing_service: PreprocessingService,
        training_service: TrainingService,
        prediction_service: PredictionService,
    ):
        super().__init__()
        self._initialized = False
        self.data_service = data_service
        self.preprocessing_service = preprocessing_service
        self.training_service = training_service
        self.prediction_service = prediction_service
        self.logger = logger["orchestration"]

    async def run_training_pipeline(
        self,
        symbol: str,
        trainer_name: str,
        start_date: str = None,
        end_date: str = None,
    ):
        # TODO What parameters + What to return ?
        run_training_pipeline()

    async def run_prediction_pipeline(self, symbol: str, model_type: str):
        # TODO What parameters + What to return ?
        run_prediction_pipeline(self.prediction_service)

    async def run_evaluation_pipeline(self, symbol: str, model_type: str):
        # TODO What parameters + What to return ?
        run_evaluation_pipeline()
