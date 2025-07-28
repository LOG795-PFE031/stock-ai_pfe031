from prefect import task
import httpx

from core.config import config

@task(
    name="production_model_exists",
    description="Check if a production model exists using the deployment service.",
    retries=3,
    retry_delay_seconds=2,
)
async def production_model_exists(
    prod_model_name: str,
) -> bool:
    """
    Check if a production model with the given name exists.

    Args:
        prod_model_name (str): Name of the production model to check.

    Returns:
        bool: True if a production model exists, False otherwise.
    """
    url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/models/{prod_model_name}/exists" 
    
    response = httpx.get(url, timeout=None)
    response.raise_for_status()
    data = response.json()

    return data["exists"]
