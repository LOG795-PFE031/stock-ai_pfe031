from prefect import task

from services import EvaluationService


@task(retries=2, retry_delay_seconds=5)
async def evaluate(
    model_name: str, true_target, pred_target, service: EvaluationService
):
    return await service.evaluate(
        model_name=model_name, y_true=true_target, y_pred=pred_target
    )


@task(retries=2, retry_delay_seconds=5)
async def should_deploy_model(
    candidate_metrics, live_metrics, service: EvaluationService
):
    return await service.should_deploy_model(candidate_metrics, live_metrics)
