from prefect import task

import httpx
from core.config import config

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
    url = f"http://{config.evaluation_service.HOST}:{config.evaluation_service.PORT}/evaluation/ready_for_deployment"
    
    payload = {
        "candidate_metrics": candidate_metrics,
        "live_metrics": live_metrics,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["ready_for_deployment"]
    
    # return await service.is_ready_for_deployment(candidate_metrics, live_metrics)
