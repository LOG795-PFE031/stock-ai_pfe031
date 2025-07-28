from prefect import task
# from services import DeploymentService
from typing import Any, Dict
import httpx
from core.config import config

@task(
    name="mlflow_log_metrics",
    description="Log evaluation metrics to MLflow using the deployment service.",
    retries=3,
    retry_delay_seconds=5,
    timeout_seconds=30,
)
async def log_metrics_to_mlflow(
    model_identifier: str,
    metrics: Dict[str, Any], # metrics: dict
    # service: DeploymentService,
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
    url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/models/{model_identifier}/log_metrics"
    
    payload = {"metrics": metrics}
    
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("logged", False)
    
    # return await service.log_metrics(model_identifier=model_identifier, metrics=metrics)
