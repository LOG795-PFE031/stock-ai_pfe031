from .evaluator import Evaluator
from .mlflow_model_manager import MLflowModelManager
from .predictor import Predictor
from services.base_service import BaseService
from core.logging import logger
from core.utils import get_model_name
from core.types import Metrics


class DeploymentService(BaseService):
    def __init__(self):
        super().__init__()
        self.logger = logger["deployment"]
        self.mlflow_model_manager = None
        self.predictor = None
        self.evaluator = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Deployment service."""
        try:
            await self._load_models()
            self.mlflow_model_manager = MLflowModelManager()
            self.predictor = Predictor()
            self.evaluator = Evaluator()
            self._initialized = True
            self.logger.info("Deployment service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize deployment service: {str(e)}")
            raise

    async def _load_models(self) -> None:
        """Load all models from disk."""
        pass

    async def evaluate(self, model_name: str, X, y) -> Metrics:
        """
        Run predictions and evaluate the model.

        Args:
            model_name (str): The name of the model to use.
            X: Input features.
            y: Ground truth.

        Returns:
            Metrics: Computed evaluation metrics.
        """
        try:
            # Prediction
            y_pred = await self.predict(model_name, X)

            # Evaluation
            metrics = self.evaluator.evaluate(y, y_pred)

            # Logs the metrics to MLFlow
            self.mlflow_model_manager.log_metrics(model_name, metrics.__dict__)

            return metrics
        except Exception as e:
            self.logger.error(f"Failed to evaluate with model {model_name} : {str(e)}")
            raise

    async def predict(self, model_name: str, X):
        """
        Run prediction on input data

        Args:
            model_name (str): The name of the model to use.
            X: Input features.

        Returns:
            Any: Predicted values from the model.
        """
        try:
            model = await self.mlflow_model_manager.load_model(model_name)
            return self.predictor.predict(model, X)
        except Exception as e:
            self.logger.error(f"Failed to predict with model {model_name} : {str(e)}")
            raise

    async def promote_model(self, model_type: str, symbol: str, X, y):
        """
        Promote a logged training model to the production model registry.

        This function evaluates the production model (if it exists) and
        promotes the training model only if it has a lower MAE.

        Parameters:
            model_type (str): The model type to promote
            symbol (str): The stock symbol
            X: Input features.
            y: Ground truth.
        """
        try:
            training_model_name = get_model_name(model_type, symbol, "training")

            if not self.mlflow_model_manager.model_exists(training_model_name):
                self.logger.info(
                    f"No training model to promote for {model_type} model for symbol {symbol}"
                )
                return False

            prod_model_name = get_model_name(model_type, symbol, "prediction")

            # Prod model Evaluation
            if self.mlflow_model_manager.model_exists(prod_model_name):
                metrics = await self.evaluate(prod_model_name, X, y)

                prod_mae = metrics.mae
                training_mae = self.mlflow_model_manager.get_metrics(
                    training_model_name
                )["mae"]

                if prod_mae < training_mae:
                    self.logger.info(
                        f"Promotion not needed: trained model has higher MAE ({training_mae}) than production model ({prod_mae})."
                    )
                    return False

            # Promotion
            self.mlflow_model_manager.promote(training_model_name, prod_model_name)
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to promote the training model for {model_type} model for symbol {symbol} : {str(e)}"
            )
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Deployment service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during deployment service cleanup: {str(e)}")
