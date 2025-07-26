from prefect import task

import httpx

from core.config import config
# from services import DeploymentService

@task(
    name="production_model_exists",
    description="Check if a production model exists using the deployment service.",
    retries=3,
    retry_delay_seconds=2,
)
async def production_model_exists(
    prod_model_name: str, 
    # service: DeploymentService
) -> bool:
    """
    Check if a production model with the given name exists.

    Args:
        prod_model_name (str): Name of the production model to check.
        service (DeploymentService): Deployment service.

    Returns:
        bool: True if a production model exists, False otherwise.
    """
    url = (
        f"http://{config.deployment_service.HOST}:"
        f"{config.deployment_service.PORT}"
        f"/deployment/models/{prod_model_name}/exists"
    )
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()
    
    # return await service.production_model_exists(prod_model_name)
