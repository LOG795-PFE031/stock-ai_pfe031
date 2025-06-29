from prefect import task

from services import DeploymentService


@task
async def predict(model_name: str, X, service: DeploymentService):
    return await service.predict(model_name, X)
