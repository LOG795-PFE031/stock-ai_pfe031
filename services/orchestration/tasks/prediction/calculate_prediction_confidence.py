from prefect import task

from typing import Any, List, Union
import numpy as np
import pandas as pd
import httpx
from core.config import config

from services import DeploymentService
from core.types import ProcessedData

@task(
    name="prediction_confidence_calculation",
    description="Calculate confidence scores for a model's predictions.",
    retries=3,
    retry_delay_seconds=5,
)
def calculate_prediction_confidence(
    model_type: str,
    symbol: str,
    y_pred,
    prediction_input,
    service: DeploymentService
) -> list[float]:
    """
    Calculate prediction confidence scores using a deployment service.

    Args:
        model_type (str): Type of model (e.g. "prophet", "lstm").
        y_pred: Model output predictions.
        prediction_input: Original input data used for predictions.
        service (DeploymentService): Service handling deployment logic.

    Returns:
        list[float]: Confidence scores
    """
    
    # Build the payload
    def to_payload(x):
        if isinstance(x, pd.DataFrame):
            return x.values.tolist()
        if isinstance(x, (pd.Series, np.ndarray)):
            return x.tolist()
        return x
    
    # Serialize the date
    def serialize_date(d):
        if d is None:
            return None
        if isinstance(d, str):
            return d
        # datetime or date
        return d.isoformat() if hasattr(d, "isoformat") else str(d)

    # Brick by bricks
    payload = {
        "model_type": model_type,
        "symbol": symbol,
        "prediction_input": {
            "X": to_payload(prediction_input.X),
            "y": to_payload(prediction_input.y),
            "feature_index_map": prediction_input.feature_index_map,
            "start_date": serialize_date(prediction_input.start_date),
            "end_date":   serialize_date(prediction_input.end_date),
        },
        "y_pred": to_payload(y_pred),
    }

    url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/calculate_prediction_confidence"
    response = httpx.post(url, json=payload, timeout=None)
    response.raise_for_status()

    return response.json().get("confidences", [])
    
    # return await service.calculate_prediction_confidence(
    #     model_type=model_type, symbol=symbol, prediction_input=prediction_input, y_pred=y_pred
    # )
