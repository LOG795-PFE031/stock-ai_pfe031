from datetime import date, datetime, timedelta
import hashlib
import json
from typing import Any, Union

import numpy as np
import pandas as pd

from core.logging import logger
from .confidence import ConfidenceCalculator
from core import BaseService
from .mlflow_model_manager import MLflowModelManager
from core.prometheus_metrics import prediction_confidence


class DeploymentService(BaseService):
    """
    Handles the deployment and management of MLflow models.

    This service supports:
    - Initialization and loading of production models.
    - Checking model existence in the MLflow registry.
    - Performing predictions.
    - Computing prediction confidence scores.
    - Logging evaluation metrics to MLflow.
    - Promoting trained models to production in the MLflow registry.
    - Managing model metadata and lifecycle.

    It integrates with MLflow, Prometheus, and supports safe error logging and resource cleanup.
    """

    def __init__(self):
        super().__init__()
        self.logger = logger["deployment"]
        self.mlflow_model_manager = None
        self.evaluator = None
        self._initialized = False
        self._prediction_cache = {}

    async def initialize(self) -> None:
        """Initialize the Deployment service."""
        try:
            self.mlflow_model_manager = MLflowModelManager()
            self._initialized = True
            self.logger.info("Deployment service initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize deployment service: %s", str(e))
            raise

    async def production_model_exists(self, prod_model_name: str) -> bool:
        """
        Checks whether a production model (live model) with the given name exists
        in the MLflow registry.

        Args:
            prod_model_name (str): The name of the production model to check.

        Returns:
            bool: True if the production model exists, False otherwise.
        """
        try:
            self.logger.info(
                "Checking if the prodcution model '%s' exists.", prod_model_name
            )
            # Get all the list of the registred (production) models corresponding to
            # the production model name
            prod_model = self.mlflow_model_manager.find_registred_model(prod_model_name)

            if prod_model:
                self.logger.info("Prodcution model '%s' exists.", prod_model_name)
                return True

            self.logger.info("Prodcution model '%s' does not exists.", prod_model_name)
            return False
        except Exception as e:
            self.logger.error(
                "Failed to check if %sproduction model exists: %s",
                prod_model_name,
                str(e),
            )
            raise

    async def list_models(self):
        """
        Retrieves and returns a list of all available production models (live models)
        with detailed information from the MLflow model registry.

        Returns:
            List[dict]: A list of dictionaries containing model details.
        """
        try:
            self.logger.info("Listing all avalaible models (in MLFlow).")
            available_models = await self.mlflow_model_manager.list_models()

            return available_models
        except Exception as e:
            self.logger.error("Failed to list the models: %s", str(e))
            raise

    async def get_model_metadata(self, model_name: str) -> dict[str, Any]:
        """
        Retrieves metadata for a specific model by its ID.

        Args:
            model_name (str): The name of the model to retrieve metadata for. ex: lstm_INTC

        Returns:
            dict[str, Any]: A dictionary containing model metadata.
        """
        try:
            self.logger.info("Retrieving metadata for model %s.", model_name)
            metadata = await self.mlflow_model_manager.get_model_metadata(model_name)
            return metadata
        except Exception as e:
            self.logger.error("Failed to get model metadata: %s", str(e))
            raise

    async def predict(
        self, model_identifier: str, X: Union[pd.DataFrame, np.ndarray, list]
    ) -> dict[str, Any]:
        """
        Run prediction on input data using either a logged model or a registered production model.

        Args:
            model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
            X: Input features.

        Returns:
            dict[str,Any]: A dictionary containing:
                - "predictions": The prediction output from the model (e.g., list, ndarray,
                    or DataFrame).
                - "model_version": The version of the model used for prediction.
        """

        try:
            self.logger.info("Starting prediction using model %s.", model_identifier)

            # Generate input hash for caching.
            input_hash = self._hash_input(X)
            if not input_hash:
                self.logger.warning(
                    "Could not generate a valid input hash. Caching will be skipped."
                )
                cache_key = None
            else:
                cache_key = (model_identifier, input_hash)
                self.logger.debug("Cache key generated: %s", cache_key)

            # Load model version
            model_result = self.mlflow_model_manager.load_model(model_identifier)
            model = model_result["model"]
            current_version = model_result["version"]

            # Log the successful loading of the model
            self.logger.debug(
                "Model %s successfully loaded with version %s.",
                model_identifier,
                current_version,
            )

            # Use cache if available and version matches
            if cache_key and cache_key in self._prediction_cache:
                cached_pred, cached_ver = self._prediction_cache[cache_key]
                if cached_ver == current_version:
                    self.logger.debug(
                        "Using cached prediction for model %s with input hash %s",
                        model_identifier,
                        input_hash,
                    )
                    return {"predictions": cached_pred, "model_version": cached_ver}

            # Perform prediction
            predictions = model.predict(X)

            # Log the completion of the prediction
            self.logger.info("Prediction completed for model %s.", model_identifier)

            # Cache the result
            if cache_key:
                self._prediction_cache[cache_key] = (predictions, current_version)
                self.logger.debug("Prediction cached for key %s.", cache_key)

            return {"predictions": predictions, "model_version": str(current_version or "unknown")} # Because sometime version is none

        except Exception as e:

            self.logger.error(
                "Failed to predict with model %s: %s", model_identifier, str(e)
            )
            raise

    async def calculate_prediction_confidence(
        self, model_type: str, symbol: str, prediction_input, y_pred
    ) -> list[float]:
        """
        Calculates the prediction confidence score for the specified model type.

        Args:
            model_type (str): The type of the model (e.g., "lstm", "prophet").
            symbol: Stock symbol. (It's to log the metric).
            prediction_input: The input data used for the prediction.
            y_pred: The model's predicted output.

        Returns:
            list[float] | None: The confidence(s) score(s) between 0 and 1, or None if unsupported.
        """
        try:
            self.logger.info(
                "Starting prediction confidence calculation with %s model.", model_type
            )
            # Confidences calculation
            confidences = ConfidenceCalculator(model_type).calculate_confidence(
                y_pred, prediction_input
            )

            # emit the gauge for each value (gauge holds last)
            if confidences is not None:
                vals = (
                    confidences
                    if isinstance(confidences, (list, tuple))
                    else [confidences]
                )
                for c in vals:
                    prediction_confidence.labels(
                        model_type=model_type, symbol=symbol
                    ).set(float(c))

            self.logger.info(
                "Prediction confidence calculation doned with %s model.", model_type
            )
            return confidences

        except Exception as e:
            self.logger.error(
                "Failed to calculate prediction confidence score with model %s : %s",
                model_type,
                str(e),
            )
            raise

    async def log_metrics(self, model_identifier: str, metrics: dict) -> bool:
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
                "Metrics successfully logged to MLflow for model %s", model_identifier
            )

            return True
        except Exception as e:
            self.logger.error(
                "Failed to log metrics to MLFlow with model %s : %s",
                model_identifier,
                str(e),
            )
            return False

    async def promote_model(self, run_id: str, prod_model_name: str) -> dict[str, Any]:
        """
        Promote a logged training model to the production model registry.

        Parameters:
            run_id (str): The run id of the logged trained model
            prod_model_name (str): Name of the production model

        Returns:
            dict[str,Any]: Deployment results
        """
        try:
            # Promotion
            mv = self.mlflow_model_manager.promote(run_id, prod_model_name)
            self.logger.info("Successfully promoted %s model to MLflow", mv.name)

            # Invalidate all cache entries for the production model name
            to_remove = [k for k in self._prediction_cache if k[0] == prod_model_name]
            for key in to_remove:
                del self._prediction_cache[key]
                self.logger.info(
                    "Invalidated prediction cache for model %s after promotion", key[0]
                )

            return {
                "deployed": True,
                "model_name": mv.name,
                "version": mv.version,
                "run_id": mv.run_id,
            }

        except Exception as e:
            self.logger.error(
                "Failed to promote the training model for %s model : %s",
                prod_model_name,
                str(e),
            )
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._initialized = False
            self.logger.info("Deployment service cleaned up successfully")
        except Exception as e:
            self.logger.error("Error during deployment service cleanup: %s", str(e))

    def _hash_input(self, X) -> str:
        """
        Returns an MD5 hash of the prediction input `X`, used for caching.

        Parameters:
            X: Prediction input data to be hashed.

        Returns:
            str: MD5 hash string or None on error.
        """
        try:

            def convert(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient="records")
                if isinstance(obj, pd.Series):
                    return obj.tolist()
                if isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                if isinstance(obj, (pd.Timestamp, datetime, date)):
                    return obj.isoformat()
                if isinstance(obj, (timedelta, pd.Timedelta)):
                    return str(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            input_str = json.dumps(X, sort_keys=True, default=convert)
            return hashlib.md5(input_str.encode()).hexdigest()
        except Exception as e:
            self.logger.error("Failed to hash input for caching: %s", str(e))
            return None
