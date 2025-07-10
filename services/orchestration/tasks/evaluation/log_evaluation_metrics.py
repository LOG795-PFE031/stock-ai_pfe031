from prefect import task
from services import DeploymentService


@task(
    name="mlflow_log_metrics",
    description="Log evaluation metrics to MLflow using the deployment service.",
    retries=3,
    retry_delay_seconds=5,
    timeout_seconds=30,
)
async def log_metrics_to_mlflow(
    model_identifier: str, metrics: dict, service: DeploymentService
) -> bool:
    """
    Log evaluation metrics to MLflow for a specific model.

    Args:
        model_identifier (str): Unique identifier for the model (e.g., run ID or model name).
        metrics (dict): Dictionary of metric names and values to log.
        service (DeploymentService): Deployment service interacting with MLflow.

    Returns:
        bool: True if the metrics were logged successfully to Mlflow.
    """
    return await service.log_metrics(model_identifier=model_identifier, metrics=metrics)
