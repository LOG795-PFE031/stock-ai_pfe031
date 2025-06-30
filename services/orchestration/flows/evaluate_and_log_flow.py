from prefect import flow
from .tasks import evaluate, log_metrics


@flow(
    name="Evaluate and logs metrics (Sub-Pipeline)", retries=2, retry_delay_seconds=10
)
async def run_evaluate_and_log_flow(
    model_name, true_target, pred_target, evaluation_service, deployment_service
):

    # Evaluate the model
    metrics = await evaluate(
        model_name=model_name,
        true_target=true_target.y,
        pred_target=pred_target.y,
        service=evaluation_service,
    )

    # Log the metrics
    await log_metrics(model_name, metrics, deployment_service)

    return metrics
