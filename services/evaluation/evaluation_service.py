from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from core.logging import logger
from core import BaseService
from .types import Metrics
from core.prometheus_metrics import (
    evaluation_mae,
    evaluation_mse,
    evaluation_rmse,
    evaluation_r2,
)


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
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: str,
        symbol: str,
    ) -> dict:
        """
        Evaluate the performance of a trained model using predicted and true target values.

        Args:
            y_true (np.ndarray): Ground truth target values.
            y_pred (np.ndarray): Predicted target values by the model.
            model_type (str): The type of model (e.g., 'lstm', 'prophet').
            symbol (str): The stock symbol (e.g., 'AAPL').

        Returns:
            Metrics: Dictionary of evaluation metrics (e.g., {'mae': ..., 'rmse': ...}).
        """
        try:
            self.logger.info(f"Starting evaluation for {model_type}_{symbol}")

            metrics_obj: Metrics = self._calculate_metrics(y_true, y_pred)
            metrics = metrics_obj.__dict__

            # Emit gauges
            evaluation_mae.labels(model_type=model_type, symbol=symbol).set(
                metrics["mae"]
            )
            evaluation_mse.labels(model_type=model_type, symbol=symbol).set(
                metrics["mse"]
            )
            evaluation_rmse.labels(model_type=model_type, symbol=symbol).set(
                metrics["rmse"]
            )
            evaluation_r2.labels(model_type=model_type, symbol=symbol).set(
                metrics["r2"]
            )
            self.logger.debug(
                f"Emitted evaluation metrics to Prometheus for {model_type}_{symbol}: {metrics}"
            )

            # Returns the metrics as a dict
            return metrics
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

    async def is_ready_for_deployment(
        self, candidate_metrics: dict, live_metrics: dict
    ) -> bool:
        """
        Determines whether the candidate model should be deployed by comparing it
        to the currently deployed (live) model.

        Args:
            candidate_metrics (Metrics): Metrics for the candidate model.
            live_metrics (Metrics): Metrics for the current live model.

        Returns:
            bool: True if the candidate should be deployed, False otherwise.
        """
        try:
            self.logger.info(f"Evaluating deployment decision based on metrics")

            # Check if deployment is needed
            deploy = self._has_better_metrics(candidate_metrics, live_metrics)

            # Log the result
            if deploy:
                self.logger.info(
                    "Candidate model performs better. Ready for deployment."
                )
            else:
                self.logger.info("Candidate model does not outperform live model.")

            return deploy
        except Exception as e:
            self.logger.error(f"Error during deployment evaluation: {e}")
            raise

    def _has_better_metrics(self, candidate_metrics: dict, live_metrics: dict) -> bool:
        """
        Returns True if the candidate model has a lower MAE than the live model.
        """
        try:
            return candidate_metrics["mae"] < live_metrics["mae"]
        except (KeyError, TypeError) as e:
            self.logger.warning(f"Missing or invalid 'mae' in metrics: {e}")
            return False

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
