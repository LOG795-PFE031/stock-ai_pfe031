from prefect import flow
from .tasks import evaluate, log_metrics


@flow(name="Evaluate and logs metrics (Sub-Pipeline)")
def run_evaluate_and_log_flow(
    model_identifier, true_target, pred_target, evaluation_service, deployment_service
):
    # Evaluate the model
    metrics_future = evaluate.submit(
        true_target=true_target.y,
        pred_target=pred_target.y,
        service=evaluation_service,
    )

    # Wait for it to finish and get result
    metrics = metrics_future.result()

    # Log the metrics
    future = log_metrics.submit(model_identifier, metrics, deployment_service)
    future.result()

    return metrics
