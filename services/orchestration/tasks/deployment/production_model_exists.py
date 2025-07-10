from prefect import task
from services import DeploymentService


@task(
    name="production_model_exists",
    description="Check if a production model exists using the deployment service.",
    retries=3,
    retry_delay_seconds=2,
)
async def production_model_exists(
    prod_model_name: str, service: DeploymentService
) -> bool:
    """
    Check if a production model with the given name exists.

    Args:
        prod_model_name (str): Name of the production model to check.
        service (DeploymentService): Deployment service.

    Returns:
        bool: True if a production model exists, False otherwise.
    """
    return await service.production_model_exists(prod_model_name)
