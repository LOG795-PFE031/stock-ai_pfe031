from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from core.logging import logger
from .base_service import BaseService
from core.types import Metrics


class EvaluationService(BaseService):

    def __init__(self):
        self.logger = logger["evaluation"]

    async def initialize(self) -> None:
        """Initialize the evaluation service."""
        try:
            self._initialized = True
            self.logger.info("Evaluation service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize evaluation service: {str(e)}")
            raise

    async def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Metrics:
        """
        Evaluate the performance of a trained model using predicted and true target values.

        Args:
            model_type (str): Type of the model being evaluated (e.g., 'lstm', 'prophet').
            y_true (np.ndarray): Ground truth target values.
            y_pred (np.ndarray): Predicted target values by the model.

        Returns:
            Metrics: Dictionary of evaluation metrics (e.g., {'mae': ..., 'rmse': ...}).
        """

        try:
            self.logger.info(f"Starting evaluation for {model_name} model")

            metrics = self._calculate_metrics(y_true, y_pred)

            return metrics
        except Exception as e:
            self.logger.error(f"Evaluation failed for {model_name} model: {str(e)}")
            raise

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return Metrics(mae=float(mae), mse=float(mse), rmse=float(rmse), r2=float(r2))

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Evaluation service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during evaluation service cleanup: {str(e)}")
