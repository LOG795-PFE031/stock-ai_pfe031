import mlflow
from mlflow import MlflowClient


class MLflowModelManager:
    def __init__(self):
        self.client = MlflowClient()

    async def list_models(self):
        """
        Lists all registered MLflow models
        """
        try:
            models = self.client.search_registered_models()
            return [model.name for model in models]
        except Exception as e:
            raise RuntimeError(f"Error listing the models: {str(e)}") from e

    async def load_model(self, model_name: str):
        """
        Load the latest MLflow model

        Args:
            model_name (str): The model name

        Returns:
            mlflow.pyfunc.PythonModel: The loaded MLflow model
        """
        try:
            # Path to the model
            model_uri = f"models:/{model_name}/latest"

            # Load the model
            model = mlflow.pyfunc.load_model(model_uri)

            return model

        except Exception as e:
            raise RuntimeError(f"Error loading the model {model_name}: {str(e)}") from e

    def log_metrics(self, model_name: str, metrics: dict):
        """
        Log evaluation metrics to MLflow for a given model.

        Args:
            model_name (str): The model name
            metrics (dict): Metrics (evaluation)
        """
        try:
            # Get the last run_id of the model
            run_id = self._get_run_id_for_model(model_name)

            # Log the metrics
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics)

        except Exception as e:
            raise RuntimeError(
                f"Failed to log metrics for model '{model_name}': {str(e)}"
            ) from e

    def promote(self, train_model_name: str, prod_model_name: str):
        try:
            # Get the last train run_id of the model
            train_run_id = self._get_run_id_for_model(train_model_name)

            # Generate the training model uri
            train_model_uri = f"runs:/{train_run_id}/{train_model_name}"

            # Promote the model (training to prediction (prod))
            mv = mlflow.register_model(train_model_uri, prod_model_name)
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
            run_id = self._get_run_id_for_model(model_name)
            run = self.client.get_run(run_id)

            # Retrieve the metrics from the run
            metrics = run.data.metrics

            return metrics
        except Exception as e:
            raise RuntimeError(
                f"Error retrieving metrics for model {model_name}: {str(e)}"
            ) from e

    def _get_run_id_for_model(self, model_name: str) -> str:
        """
        Retrieve the run ID for the latest registered MLflow model version

        Args:
            model_name (str): The model name

        Returns:
            str: The run ID associated with the latest version of the model.
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")

            if not versions:
                raise RuntimeError(f"No versions found for model '{model_name}'.")

            # Sort by version number (descending, most recent first)
            sorted_versions = sorted(
                versions, key=lambda v: int(v.version), reverse=True
            )

            return sorted_versions[0].run_id

        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve run ID for model '{model_name}': {str(e)}"
            ) from e
