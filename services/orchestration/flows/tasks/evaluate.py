from prefect import task
from services import DeploymentService


@task
async def evalaute(
    symbol: str, model_type: str, training_data, service: DeploymentService
):
    pass
