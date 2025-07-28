from prefect import task

from typing import Any, List, Union
import numpy as np
import pandas as pd
import httpx
from core.config import config

from services import DeploymentService


@task(
    name="prediction_confidence_calculation",
    description="Calculate confidence scores for a model's predictions.",
    retries=3,
    retry_delay_seconds=5,
)
async def calculate_prediction_confidence(
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
    
    # Serialize inputs
    # def to_payload(x):
    #     if isinstance(x, pd.DataFrame):
    #         return x.values.tolist()
    #     if isinstance(x, np.ndarray):
    #         return x.tolist()
    #     return x

    # payload = {
    #     "model_type": model_type,
    #     "symbol": symbol,
    #     "prediction_input": to_payload(prediction_input),
    #     "y_pred": to_payload(y_pred),
    # }

    # url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/calculate_prediction_confidence"
    # async with httpx.AsyncClient(timeout=None) as client:
    #     resp = await client.post(url, json=payload)
    #     resp.raise_for_status()
    #     data = resp.json() 
    #     return data.get("confidences", [])
    
    
    return await service.calculate_prediction_confidence(
        model_type=model_type, symbol=symbol, prediction_input=prediction_input, y_pred=y_pred
    )
