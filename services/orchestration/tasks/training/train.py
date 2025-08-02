from typing import Any
import httpx
from prefect import task
import numpy as np
import pandas as pd

from core.config import config
from ...types import ProcessedData


@task(
    name="model_training",
    description="Train a model for a given symbol using training data.",
    retries=2,
    retry_delay_seconds=5,
)
async def train(
    symbol: str, model_type: str, training_data: ProcessedData
) -> dict[str, Any]:
    """
    Train a model for a given symbol using training data.

    Args:
        symbol (str): Stock ticker symbol.
        model_type (str): Type of model (e.g. "prophet", "lstm").
        training_data: The data used for training the model.

    Returns:
        dict[str,Any]: Training infos (containing the `run_id` key to locate the training model)
    """

    # Form the url to the training endpoint of the training service
    url = f"http://{config.training_service.HOST}:{config.training_service.PORT}/training/train"

    # TODO Toute la logique de regarder les instances ne devraient pas se passer
    # il faut continuer le preprocessing service
    is_numpy_instance = isinstance(training_data.X, np.ndarray)
    is_dataframe = hasattr(training_data.X, "to_dict")

    # Build X payload safely
    if is_numpy_instance:
        X_payload = training_data.X.tolist()
    elif is_dataframe:
        X_payload = training_data.X.to_dict(orient="records")
    else:
        X_payload = training_data.X  # Fallback: already JSON-compatible

    # Define the payload
    payload = {
        "data": {
            "X": X_payload,
            "y": (
                training_data.y.tolist()
                if isinstance(training_data.y, np.ndarray)
                or isinstance(training_data.y, pd.Series)
                else training_data.y
            ),
            "feature_index_map": training_data.feature_index_map,
            "start_date": training_data.start_date,
            "end_date": training_data.end_date,
        },
    }

    # Define the query parameters
    params = {
        "symbol": symbol,
        "model_type": model_type,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        # Send POST request to FastAPI endpoint
        response = await client.post(url, params=params, json=payload)

        # Check if the response is successful
        response.raise_for_status()

        return response.json()
