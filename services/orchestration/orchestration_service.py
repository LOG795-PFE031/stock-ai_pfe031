from datetime import datetime

from core.logging import logger
from core.utils import format_prediction_response
from .flows import (
    run_evaluation_pipeline,
    run_prediction_pipeline,
    run_training_pipeline,
)
from ..base_service import BaseService
from ..deployment import DeploymentService
from ..training import TrainingService
from ..data_processing import DataProcessingService
from ..evaluation_service import EvaluationService
from ..data_service import DataService
from core.logging import logger


class OrchestrationService(BaseService):

    def __init__(
        self,
        data_service: DataService,
        preprocessing_service: DataProcessingService,
        training_service: TrainingService,
        deployment_service: DeploymentService,
        evaluation_service: EvaluationService,
    ):
        super().__init__()
        self.data_service = data_service
        self.preprocessing_service = preprocessing_service
        self.training_service = training_service
        self.deployment_service = deployment_service
        self.evaluation_service = evaluation_service
        self.logger = logger["orchestration"]

    async def initialize(self) -> None:
        """Initialize the orchestration service."""
        try:
            self._initialized = True
            self.logger.info("Orchestration service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestration service: {str(e)}")
            raise

    async def run_training_pipeline(self, model_type: str, symbol: str):
        """
        Run the full training pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being trained.

        Returns:
            result: The result of the training pipeline execution.
        """

        try:
            # Enforce upper case for the symbol
            symbol = symbol.upper()

            self.logger.info(
                f"Starting training pipeline for {model_type} model for {symbol}."
            )

            # Run the training pipeline
            result = run_training_pipeline(
                model_type,
                symbol,
                self.data_service,
                self.preprocessing_service,
                self.training_service,
                self.deployment_service,
                self.evaluation_service,
            )

            self.logger.info(
                f"Training pipeline completed successfully for {model_type} model for {symbol}."
            )

            # Add a success status to the result
            result["status"] = "success"

            return result
        except Exception as e:
            self.logger.error(
                f"Error running the training pipeline for model {model_type} for {symbol}: {str(e)}"
            )
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def run_prediction_pipeline(self, model_type: str, symbol: str):
        """
        Run the full prediction pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being used for prediction.

        Returns:
            result: The result of the prediction pipeline execution.
        """
        try:
            # Enforce upper case for the symbol
            symbol = symbol.upper()

            self.logger.info(
                f"Starting prediction pipeline for {model_type} model for {symbol}."
            )

            # Run the prediction pipeline
            prediction_result = run_prediction_pipeline(
                model_type,
                symbol,
                self.data_service,
                self.preprocessing_service,
                self.deployment_service,
            )

            if prediction_result:

                # Extract the prediction results
                prediction = prediction_result["y_pred"][0]
                confidence = prediction_result["confidence"][0]
                model_version = prediction_result["model_version"]

                self.logger.info(
                    f"Prediction pipeline completed successfully for {model_type} model for {symbol}."
                )

                return format_prediction_response(
                    prediction=prediction,
                    confidence=confidence,
                    model_type=model_type,
                    symbol=symbol,
                    model_version=model_version,
                )

            else:
                self.logger.info(
                    f"No live model available to make prediction with {model_type} model for {symbol}."
                )
                return {
                    "status": "error",
                    "error": f"No live model available to make prediction with {model_type} model for {symbol}.",
                    "symbol": symbol,
                    "model_type": model_type,
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            self.logger.error(
                f"Error running the prediction pipeline for model {model_type} for {symbol}: {str(e)}"
            )
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def run_evaluation_pipeline(self, model_type: str, symbol: str):
        """
        Run the full evaluation pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being evaluated.

        Returns:
            result: The result of the evaluation pipeline execution.
        """

        try:
            # Enforce upper case for the symbol
            symbol = symbol.upper()

            self.logger.info(
                f"Starting evaluation pipeline for {model_type} model for {symbol}."
            )

            # Run the evaluation pipeline
            result = run_evaluation_pipeline(
                model_type,
                symbol,
                self.data_service,
                self.preprocessing_service,
                self.deployment_service,
                self.evaluation_service,
            )

            # Log the successful completion of the pipeline
            self.logger.info(
                f"Evaluation pipeline completed successfully for {model_type} model for {symbol}."
            )

            return result
        except Exception as e:
            self.logger.error(
                f"Error running the evaluation pipeline for model {model_type} for {symbol}: {str(e)}"
            )
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Orchestration service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during orchestration service cleanup: {str(e)}")
