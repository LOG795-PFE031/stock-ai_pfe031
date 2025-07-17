from prefect import task
from typing import Any

from services import DeploymentService


@task(
    name="promote_model",
    description="Promote a trained model to production using the deployment service.",
    retries=3,
    retry_delay_seconds=5,
)
async def promote_model(
    run_id: str,
    prod_model_name: str,
    service: DeploymentService,
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
    return await service.promote_model(run_id=run_id, prod_model_name=prod_model_name)
