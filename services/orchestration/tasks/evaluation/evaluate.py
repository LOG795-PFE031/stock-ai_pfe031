from prefect import task

import httpx
from core.config import config
import numpy as np

@task(
    name="model_evaluation",
    description="Evaluate model predictions using the evaluation service.",
    retries=2,
    retry_delay_seconds=5,
)
async def evaluate(
    true_target, 
    pred_target, 
    model_type: str, 
    symbol: str, 
) -> dict[str, float]:
    """
    Evaluate model predictions using ground truth values.

    Args:
        true_target: Ground truth target values.
        pred_target: Predicted values from the model.
        model_type: Type of the model. 
        symbol: Stock symbol. 
        service (EvaluationService): Evaluation service.

    Returns:
        dict[str,float]: Dictionary of evaluation metrics (e.g., rmse, r2, etc).
    """
    url = f"http://{config.evaluation_service.HOST}:{config.evaluation_service.PORT}/evaluation/models/{model_type}/{symbol}/evaluate"
    
    def _to_serializable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, list):
            return [_to_serializable(i) for i in o]
        return o
    
    payload = {
        "true_target": _to_serializable(true_target),
        "pred_target": _to_serializable(pred_target),
    }

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
