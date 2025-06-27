from prefect import task
from services import DeploymentService


@task
async def evaluate(model_name: str, X, y, service: DeploymentService):
    return await service.evaluate(model_name, X, y)
