import asyncio
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .schemas import ModelMlflowInfo, ModelVersionInfo


class MLflowModelManager:
    """
    A utility class to manage MLflow models, versions, and metrics.

    This class provides a high-level interface to interact with MLflow's tracking
    and model registry features. It supports operations such as listing models,
    loading and caching production or training models, logging evaluation metrics,
    promoting trained models to production, and retrieving model metadata.

    Attributes:
        models_cache (dict): A shared cache of loaded models.
        PRODUCTION_ALIAS (str): The alias used to identify production models.
        client (MlflowClient): MLflow client instance for registry and tracking operations.
    """

    # Loaded models cache
    models_cache = {}

    # Production alias (used to identify the production model)
    PRODUCTION_ALIAS = "production"

    def __init__(self):
        self.client = MlflowClient()

    async def list_models(self) -> list:
        """
        Retrieve a list of all registered MLflow models along with their metadata.

        This method queries the MLflow Model Registry and returns detailed information
        about each registered model, including its name, description, creation and update
        timestamps, aliases, tags, and metadata about its latest versions.

        Returns:
            list[dict]: A list of dictionaries, each containing metadata for a registered model.
        """
        try:
            # Run the blocking search in a background thread
            models = await asyncio.to_thread(
                self.client.search_registered_models, max_results=500
            )
            return [
                {
                    "name": model.name,
                    "description": model.description,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "aliases": model.aliases,
                    "tags": {tag.key: tag.value for tag in model.tags},
                    "latest_versions": [
                        {
                            "version": str(v.version),  # <-- fix here
                            "stage": v.current_stage,
                            "status": v.status,
                            "run_id": v.run_id,
                            "creation_timestamp": v.creation_timestamp,
                            "last_updated_timestamp": v.last_updated_timestamp,
                        }
                        for v in model.latest_versions
                    ],
                }
                for model in models
            ]
        except Exception as e:
            raise RuntimeError(f"Error listing the models: {str(e)}") from e

    async def get_model_metadata(self, model_name: str):
        """
        Retrieve detailed metadata for a specific MLflow model, including latest versions.

        Args:
            model_name (str): The name of the model to retrieve metadata for.

        Returns:
            ModelMlflowInfo: An object containing the model metadata.
        """
        try:
            model = self.client.get_registered_model(model_name)
            latest_versions = [
                ModelVersionInfo(
                    version=str(v.version),
                    stage=v.current_stage,
                    status=v.status,
                    run_id=v.run_id,
                    creation_timestamp=v.creation_timestamp,
                    last_updated_timestamp=v.last_updated_timestamp,
                )
                for v in model.latest_versions
            ]
            return ModelMlflowInfo(
                name=model.name,
                description=model.description,
                creation_timestamp=model.creation_timestamp,
                last_updated_timestamp=model.last_updated_timestamp,
                tags={tag.key: tag.value for tag in model.tags},
                aliases={alias: v.version for alias, v in zip(model.aliases, model.latest_versions)}, # Map them
                latest_versions=latest_versions,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error retrieving metadata for model '{model_name}': {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error retrieving metadata for model '{model_name}': {str(e)}"
            ) from e

    def find_registred_model(self, prod_model_name: str) -> list:
        """
        Find if a registered model exists by name.

        Args:
            prod_model_name (str): The name of the registered model to look for.

        Returns:
            RegisteredModel object if found, None otherwise.
        """
        try:
            model = self.client.get_registered_model(prod_model_name)
            return model
        except MlflowException as e:
            if "not found" in str(e):
                return None  # Model does not exist
            else:
                raise RuntimeError(f"Error retrieving model: {str(e)}") from e

    def load_model(self, model_identifier: str) -> dict[str, Any]:
        """
        Load the latest MLflow model

        Args:
            model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).

        Returns:
            mlflow.pyfunc.PythonModel: The loaded MLflow model
            int: Version of the loaded MLflow model
        """
        try:
            if model_identifier not in self.models_cache:
                version = None

                if self._run_exists(model_identifier):
                    # Path to the logged trained model
                    model_uri = f"runs:/{model_identifier}/model"
                else:
                    # Path to the production model (live model)
                    model_uri = f"models:/{model_identifier}@{self.PRODUCTION_ALIAS}"

                    # Get the version of the registred model
                    versions = self.client.search_model_versions(
                        f"name='{model_identifier}'"
                    )

                    for v in versions:

                        detailed = self.client.get_model_version(
                            name=model_identifier, version=v.version
                        )

                        if "production" in detailed.aliases:
                            version = v.version
                            break

                # Load the model
                model = mlflow.pyfunc.load_model(model_uri)

                # Store it to cache
                self.models_cache[model_identifier] = {
                    "model": model,
                    "version": version,
                }

            # Return cache
            return self.models_cache[model_identifier]

        except Exception as e:
            raise RuntimeError(
                f"Error loading the model {model_identifier}: {str(e)}"
            ) from e

    def log_metrics(self, model_identifier: str, metrics: dict):
        """
        Log evaluation metrics to MLflow for a given model.

        Args:
            model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
            metrics (dict): Metrics (evaluation)
        """
        try:
            # Get the last run_id of the model
            run_id = self._get_run_id_model(model_identifier)

            # Get the current step
            current_step = self._get_current_step(run_id=run_id)

            # New step
            new_step = current_step + 1

            # Log the metrics
            with mlflow.start_run(run_id=run_id):
                for key in metrics:
                    mlflow.log_metric(key, value=metrics[key], step=new_step)

            # Update the step
            self._update_step_tag(run_id=run_id, step=new_step)

        except Exception as e:
            raise RuntimeError(
                f"Failed to log metrics for model '{model_identifier}': {str(e)}"
            ) from e

    def promote(self, train_run_id: str, prod_model_name: str):
        """
        Promote a trained MLflow model (logged in a run) to a registered production model.

        Args:
            train_run_id (str): The MLflow run ID where the trained model is logged.
            prod_model_name (str): The name to register the model under in the MLflow model
                registry.

        Returns:
            ModelVersion: The registered MLflow ModelVersion object.
        """
        try:
            # Generate the training model uri
            train_model_uri = f"runs:/{train_run_id}/model"

            # Promote the model (training to prediction (prod))
            mv = mlflow.register_model(train_model_uri, prod_model_name)

            # Delete cache entry if the model was in the cached
            if prod_model_name in self.models_cache:
                del self.models_cache[prod_model_name]

            # Add the alias production to the model
            self.client.set_registered_model_alias(
                name=prod_model_name, alias=self.PRODUCTION_ALIAS, version=mv.version
            )

            # Initialize the step counter (for logging)
            self._update_step_tag(train_run_id, 0)

            return mv
        except Exception as e:
            raise RuntimeError(f"Error promoting training model : {str(e)}") from e

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in the MLflow registry.

        Args:
            model_name (str): The name of the model to check in the registry.

        Returns:
            bool: True if the model exists, False otherwise.
        """

        versions = self.client.get_latest_versions(model_name)
        return len(versions) > 0

    def get_metrics(self, model_name: str):
        """
        Retrieve metrics for a model based on its run ID.

        Args:
            model_name (str): The name of the model to retrieve metrics for.

        Returns:
            dict: A dictionary containing the metrics for the specified model.
        """
        try:
            # Get the last run of the model
            run_id = self._get_run_id_model(model_name)
            run = self.client.get_run(run_id)

            # Retrieve the metrics from the run
            metrics = run.data.metrics

            return metrics
        except Exception as e:
            raise RuntimeError(
                f"Error retrieving metrics for model {model_name}: {str(e)}"
            ) from e

    def _get_current_step(self, run_id: str, key: str = "step") -> int:
        """
        Retrieve the current step value stored as a tag in an MLflow run.

        Args:
            run_id (str): The MLflow run ID to retrieve the tag from.
            key (str): The tag key used to store the step value. Defaults to "step".

        Returns:
            int: The current step value. Returns 0 if the tag is not set.
        """
        run = self.client.get_run(run_id)
        step_str = run.data.tags.get(key, "0")
        return int(step_str)

    def _update_step_tag(self, run_id: str, step: int, key: str = "step"):
        """
        Update or create a tag in an MLflow run to store the current step value.

        Args:
            run_id (str): The MLflow run ID where the tag should be set.
            step (int): The step value to store.
            key (str): The tag key used to store the step value. Defaults to "step".
        """
        self.client.set_tag(run_id, key, str(step))

    def _get_run_id_model(self, model_identifier: str) -> str:
        """
        Retrieve the run ID of a MLflow model (live model) based on the model indentifier

        Args:
            model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of aregistered model (live model)).

        Returns:
            str: The run ID associated with the production MLflow model
        """
        try:
            if self._run_exists(model_identifier):
                return model_identifier

            versions = self.client.search_model_versions(f"name='{model_identifier}'")

            if not versions:
                raise RuntimeError(f"No versions found for model '{model_identifier}'.")

            for v in versions:
                if self.PRODUCTION_ALIAS in v.aliases:
                    return v.run_id

            raise RuntimeError(
                f"No model '{model_identifier}' has alias '{self.PRODUCTION_ALIAS}'."
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve run ID for model '{model_identifier}': {str(e)}"
            ) from e

    def _run_exists(self, run_id: str) -> bool:
        """
        Check whether a run with the given run ID exists in the MLflow tracking server.

        Args:
            run_id (str): The unique identifier of the MLflow run.

        Returns:
            bool: True if the run exists and is accessible; False if it does not exist
                or has been deleted.
        """
        try:
            mlflow.get_run(run_id)
            return True
        except Exception:
            return False
