from prefect import task

from services import DeploymentService


@task(retries=3, retry_delay_seconds=5)
async def promote_model(
    run_id: str,
    model_name: str,
    service: DeploymentService,
) -> bool:
    return await service.promote_model(run_id=run_id, prod_model_name=model_name)


@task(retries=3, retry_delay_seconds=2)
async def model_exist(model_name: str, service: DeploymentService) -> bool:
    return await service.model_exists(model_name)


@task(retries=3, retry_delay_seconds=5)
async def log_metrics(model_identifier: str, metrics, service: DeploymentService):
    return await service.log_metrics(model_identifier=model_identifier, metrics=metrics)


@task(retries=3, retry_delay_seconds=5)
async def calculate_prediction_confidence(
    model_type: str, y_pred, prediction_input, service: DeploymentService
):
    return await service.calculate_prediction_confidence(
        model_type=model_type, prediction_input=prediction_input, y_pred=y_pred
    )
