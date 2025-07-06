from prefect import flow
from .tasks import evaluate, log_metrics


@flow(name="Evaluate and logs metrics (Sub-Pipeline)")
async def run_evaluate_and_log_flow(
    model_identifier, true_target, pred_target, evaluation_service, deployment_service
):

    # Evaluate the model
    metrics = await evaluate(
        true_target=true_target.y,
        pred_target=pred_target.y,
        service=evaluation_service,
    )

    # Log the metrics
    await log_metrics(model_identifier, metrics, deployment_service)

    return metrics
