from prefect import task
from typing import Any
import httpx

from core.config import config
# from services import DeploymentService


@task(
    name="promote_model",
    description="Promote a trained model to production using the deployment service.",
    retries=3,
    retry_delay_seconds=5,
)
async def promote_model(
    run_id: str,
    prod_model_name: str,
    # service: DeploymentService,
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
    url = (
        f"http://{config.deployment_service.HOST}:"
        f"{config.deployment_service.PORT}"
        f"/deployment/models/{prod_model_name}/promote"
    )
    
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json={"run_id": run_id})
        r.raise_for_status()
        return r.json()
    
    # return await service.promote_model(run_id=run_id, prod_model_name=prod_model_name)
