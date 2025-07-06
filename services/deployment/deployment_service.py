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
        Checks whether a production model (live model) with the given name exists
        in the MLflow registry.

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
        Retrieves and returns a list of all available production model (live model) names
        from the MLflow model registry.

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

    async def predict(self, model_identifier: str, X):
        """
        Run prediction on input data using either a logged model or a registered production model.

        Args:
            model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
            X: Input features.

        Returns:
            Any: Predicted values from the model.
            int: Version of the model
        """
        try:
            self.logger.info(f"Starting prediction using model {model_identifier}.")

            # Load the model
            model, version = await self.mlflow_model_manager.load_model(
                model_identifier
            )

            # Log the successful loading of the model
            self.logger.debug(f"Model {model_identifier} successfully loaded.")

            # Perform prediction
            predictions = model.predict(X)

            # Log the completion of the prediction
            self.logger.info(f"Prediction completed for model {model_identifier}.")

            return predictions, version

        except Exception as e:
            self.logger.error(
                f"Failed to predict with model {model_identifier} : {str(e)}"
            )
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

    async def log_metrics(self, model_identifier: str, metrics: dict):
        """
        Logs evaluation metrics to MLflow for the specified model.

        Args:
            model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
            metrics: An object containing evaluation metrics to log.

        Returns:
            bool: True if the metrics were logged successfully.
        """
        try:
            self.mlflow_model_manager.log_metrics(model_identifier, metrics)
            self.logger.info(
                f"Metrics successfully logged to MLflow for model {model_identifier}"
            )

            return True
        except Exception as e:
            self.logger.error(
                f"Failed to log metrics to MLFlow with model {model_identifier} : {str(e)}"
            )
            raise

    async def promote_model(self, run_id: str, prod_model_name: str):
        """
        Promote a logged training model to the production model registry.

        Parameters:
            run_id (str): The run id of the logged trained model
            prod_model_name (str): Name of the production model
        """
        try:
            # Promotion
            mv = self.mlflow_model_manager.promote(run_id, prod_model_name)
            self.logger.info(f"Successfully promoted {mv.name} model to MLflow")

            return {
                "deployed": True,
                "model_name": mv.name,
                "version": mv.version,
                "run_id": mv.run_id,
            }

        except Exception as e:
            self.logger.error(
                f"Failed to promote the training model for {prod_model_name} model : {str(e)}"
            )
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Deployment service cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during deployment service cleanup: {str(e)}")
