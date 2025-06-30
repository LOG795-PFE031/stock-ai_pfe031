from prefect import task

from services import DeploymentService


@task
async def promote_model(
    model_type: str,
    symbol: str,
    service: DeploymentService,
) -> bool:
    return await service.promote_model(model_type=model_type, symbol=symbol)


@task
async def model_exist(model_name: str, service: DeploymentService) -> bool:
    return await service.model_exists(model_name)


@task
async def log_metrics(model_name: str, metrics, service: DeploymentService):
    return await service.log_metrics(model_name=model_name, metrics=metrics)
