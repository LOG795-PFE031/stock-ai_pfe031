from .mlflow_model_manager import MLflowModelManager
from .confidence import ConfidenceCalculator
from services.base_service import BaseService
from core.logging import logger
from core.utils import get_model_name


class DeploymentService(BaseService):
    def __init__(self):
        super().__init__()
        self.logger = logger["deployment"]
        self.mlflow_model_manager = None
        self.evaluator = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Deployment service."""
        try:
            await self._load_models()
            self.mlflow_model_manager = MLflowModelManager()
            self._initialized = True
            self.logger.info("Deployment service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize deployment service: {str(e)}")
            raise

    async def _load_models(self) -> None:
        """Load all models from disk."""
        pass

    async def model_exists(self, model_name: str) -> bool:
        """
        Checks whether a model with the given name exists in the MLflow registry.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        try:
            models = await self.list_models()
            return model_name in models
        except Exception as e:
            self.logger.error(f"Failed to check if {model_name} model exists: {str(e)}")
            raise

    async def list_models(self):
        """
        Retrieves and returns a list of all available model names from the MLflow model registry.

        Returns:
            List[str]: A list of model names
        """
        try:
            self.logger.info("Listing all avalaible models (in MLFlow).")
            available_models_names = await self.mlflow_model_manager.list_models()

            return available_models_names
        except Exception as e:
            self.logger.error(f"Failed to list the models: {str(e)}")
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
            self.logger.info(f"Starting prediction using model {model_name}.")

            # Load the model
            model = await self.mlflow_model_manager.load_model(model_name)

            # Log the successful loading of the model
            self.logger.debug(f"Model {model_name} successfully loaded.")

            # Perform prediction
            predictions = model.predict(X)

            # Log the completion of the prediction
            self.logger.info(f"Prediction completed for model {model_name}.")

            return predictions

        except Exception as e:
            self.logger.error(f"Failed to predict with model {model_name} : {str(e)}")
            raise

    async def calculate_prediction_confidence(
        self, model_type: str, prediction_input, y_pred
    ):
        """
        Calculates the prediction confidence score for the specified model type.

        Args:
            model_type (str): The type of the model (e.g., "lstm", "prophet").
            prediction_input: The input data used for the prediction.
            y_pred: The model's predicted output.

        Returns:
            float | None: The confidence score between 0 and 1, or None if unsupported.
        """
        try:
            self.logger.info(
                f"Starting prediction confidence calculation with {model_type} model."
            )
            # Confidence calculation
            confidence = ConfidenceCalculator(model_type).calculate_confidence(
                y_pred, prediction_input
            )

            self.logger.info(
                f"Prediction confidence calculation doned with {model_type} model."
            )
            return confidence

        except Exception as e:
            self.logger.error(
                f"Failed to calculate prediction confidence score with model {model_type} : {str(e)}"
            )
            raise

    async def log_metrics(self, model_name: str, metrics):
        """
        Logs evaluation metrics to MLflow for the specified model.

        Args:
            model_name (str): The name of the model in MLflow.
            metrics: An object containing evaluation metrics to log.

        Returns:
            bool: True if the metrics were logged successfully.
        """
        try:
            self.mlflow_model_manager.log_metrics(model_name, metrics.__dict__)
            self.logger.info(
                f"Metrics successfully logged to MLflow for model {model_name}"
            )

            return True
        except Exception as e:
            self.logger.error(
                f"Failed to log metrics to MLFlow with model {model_name} : {str(e)}"
            )
            raise

    async def promote_model(self, model_type: str, symbol: str):
        """
        Promote a logged training model to the production model registry.

        This function evaluates the production model (if it exists) and
        promotes the training model only if it has a lower MAE.

        Parameters:
            model_type (str): The model type to promote
            symbol (str): The stock symbol
        """
        try:
            training_model_name = get_model_name(model_type, symbol, "training")
            prod_model_name = get_model_name(model_type, symbol, "prediction")

            # Promotion
            mv = self.mlflow_model_manager.promote(training_model_name, prod_model_name)
            self.logger.info(f"Successfully promoted {mv.name} model to MLflow")

            return {
                "deployed": True,
                "model_name": mv.name,
                "version": mv.version,
                "run_id": mv.run_id,
            }

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
