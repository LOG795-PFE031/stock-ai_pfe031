from prefect import task

from services import EvaluationService


@task
async def evaluate(
    model_name: str, true_target, pred_target, service: EvaluationService
):
    return await service.evaluate(
        model_name=model_name, y_true=true_target, y_pred=pred_target
    )


@task
async def should_deploy_model(
    candidate_metrics, live_metrics, service: EvaluationService
):
    return await service.should_deploy_model(candidate_metrics, live_metrics)
