from prefect import task

from services import DeploymentService


@task(retries=3, retry_delay_seconds=5)
async def predict(model_name: str, X, service: DeploymentService):
    return await service.predict(model_name, X)
