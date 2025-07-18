import mlflow
from mlflow import MlflowClient

from api.schemas import ModelMlflowInfo, ModelVersionInfo


class MLflowModelManager:

    # Production alias (used to identify the production model)
    PRODUCTION_ALIAS = "production"

    def __init__(self):
        self.client = MlflowClient()

    async def list_models(self):
        try:
            models = self.client.search_registered_models(max_results=500)
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

    async def find_registred_model(self, prod_model_name: str) -> list:
        """
        Find if a production model exists

        Args:
            prod_model_name (str): Production model name to check
        """
        try:
            models = self.client.search_registered_models(
                filter_string=f"name='{prod_model_name}'"
            )

            return models
        except Exception as e:
            raise RuntimeError(f"Error listing the models: {str(e)}") from e

    async def load_model(self, model_identifier: str):
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
            if self._run_exists(model_identifier):
                # Path to the logged trained model
                model_uri = f"runs:/{model_identifier}/model"

                # There is no model version for a logged model
                version = None
            else:
                # Path to the production model (live model)
                model_uri = f"models:/{model_identifier}@{self.PRODUCTION_ALIAS}"

                # Get the version of the registred model
                versions = self.client.search_model_versions(
                    f"name='{model_identifier}'"
                )
                for v in versions:
                    if self.PRODUCTION_ALIAS in v.aliases:
                        version = v.version
                        break

            # Load the model
            model = mlflow.pyfunc.load_model(model_uri)

            return model, version

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
            prod_model_name (str): The name to register the model under in the MLflow model registry.

        Returns:
            ModelVersion: The registered MLflow ModelVersion object.
        """
        try:
            # Generate the training model uri
            train_model_uri = f"runs:/{train_run_id}/model"

            # Promote the model (training to prediction (prod))
            mv = mlflow.register_model(train_model_uri, prod_model_name)

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
