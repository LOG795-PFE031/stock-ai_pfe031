from core.logging import logger
from services import BaseService

from .flows import (
    run_evaluation_pipeline,
    run_prediction_pipeline,
    run_training_pipeline,
    run_data_pipeline,
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

    async def initialize(self) -> None:
        """Initialize the orchestration service."""
        try:
            self._initialized = True
            self.logger.info("Orchestration service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestration service: {str(e)}")
            raise

    async def run_data_pipeline(self, symbol: str, model_type: str, phase: str):
        return await run_data_pipeline(
            symbol, model_type, phase, self.data_service, self.preprocessing_service
        )

    async def run_training_pipeline(
        self,
        model_type: str,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
    ):
        # TODO Complete this :
        return await run_training_pipeline(
            model_type,
            symbol,
            self.data_service,
            self.preprocessing_service,
            self.training_service,
            None,
        )

    async def run_prediction_pipeline(self, model_type: str, symbol: str):
        # TODO Complete this :
        return await run_prediction_pipeline(
            model_type, symbol, self.data_service, self.preprocessing_service
        )

    async def run_evaluation_pipeline(self, model_type: str, symbol: str):
        # TODO Complete this :
        return await run_evaluation_pipeline(
            model_type, symbol, self.data_service, self.preprocessing_service, None
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Preprocessing service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during data service cleanup: {str(e)}")
