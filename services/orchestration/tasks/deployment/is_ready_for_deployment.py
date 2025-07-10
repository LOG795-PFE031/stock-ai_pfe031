from prefect import task
from services import EvaluationService


@task(
    name="is_ready_for_deployment",
    description="Evaluate whether the candidate model is ready for deployment based on comparison with live model metrics.",
    retries=2,
    retry_delay_seconds=5,
)
async def is_ready_for_deployment(
    candidate_metrics, live_metrics, service: EvaluationService
) -> bool:
    """
    Determine if the candidate model is ready for deployment by comparing evaluation metrics.

    Args:
        candidate_metrics: Evaluation metrics of the candidate model
        live_metrics: Evaluation metrics of the live model currently in production.
        service (EvaluationService): Service that handles the logic of comparing metrics.

    Returns:
        bool: True if the candidate model is ready for deployment, False otherwise.
    """
    return await service.is_ready_for_deployment(candidate_metrics, live_metrics)
