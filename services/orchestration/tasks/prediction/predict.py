from prefect import task
from typing import Any, Union, Dict
import numpy as np
import pandas as pd
import httpx

from core.config import config
# from services import DeploymentService


@task(
    name="model_prediction",
    description="Make a prediction using a specified MLFlow model and input data.",
    retries=3,
    retry_delay_seconds=5,
)
async def predict(
    model_identifier: str,
    X: Union[pd.DataFrame, np.ndarray, list],
    # service: DeploymentService,
) -> Dict[str, Any]:
    """
    Make predictions using a MLFlow model and input data.

    Args:
        model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
        X: Input data for prediction.
        service (DeploymentService): Service that handles the model prediction.

    Returns:
        Prediction result.
    """
    # Build X payload safely
    if isinstance(X, np.ndarray):
        X_payload = X.tolist()
    elif hasattr(X, "to_dict"):
        X_payload = X.to_dict(orient="records")
    else:
        X_payload = X

    url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/predict"
    payload = {
        "model_identifier": model_identifier,
        "X": X_payload,
    }

    async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

    predictions = result.get("predictions")
    if predictions is not None and isinstance(predictions, list):
        result["predictions"] = np.array(predictions)

    return result

# return await service.predict(model_identifier, X)