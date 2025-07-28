from prefect import task
from typing import Any
import httpx

from core.config import config

@task(
    name="promote_model",
    description="Promote a trained model to production using the deployment service.",
    retries=3,
    retry_delay_seconds=5,
)
async def promote_model(
    run_id: str,
    prod_model_name: str,
) -> dict[str, Any]:
    """
    Promote a trained model version to production.

    Args:
        run_id (str): Mlflow run id of the model training run.
        model_name (str): Prodcuction model name (name that the model going to have).
        service (DeploymentService): Service handling model deployment

    Returns:
        bool: True if the promotion was successful, False otherwise.
    """
    url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/models/{prod_model_name}/promote"
    
    payload = {"run_id": run_id}
    
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
