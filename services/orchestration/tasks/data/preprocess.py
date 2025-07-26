from typing import Union

import httpx
import pandas as pd
from prefect import task

from core.config import config
from core.types import ProcessedData
from ...utils import to_processed_data


@task(
    name="preprocess",
    description="Preprocess stock data for a given symbol and model type using a data processing service.",
    retries=2,
    retry_delay_seconds=5,
)
async def preprocess_data(
    symbol: str,
    data: pd.DataFrame,
    model_type: str,
    phase: str,
) -> Union[ProcessedData, tuple[ProcessedData, ProcessedData]]:
    """
    Prefect task to preprocess raw stock data for model consumption.

    Args:
        symbol (str): The stock ticker symbol to preprocess data for.
        data (pd.DataFrame): Raw stock data to preprocess.
        model_type (str): The type of model (e.g., "lstm", "prophet")
        phase (str): Phase in the pipeline (e.g. "training", "prediction", "evaluation").

    Returns:
        ProcessedData: The preprocessed data
    """

    # Get the endpoint URL
    url = (
        f"http://{config.data_processing_service.HOST}:"
        f"{config.data_processing_service.PORT}"
        "/processing/preprocess"
    )

    # Convert the dates into strings (JSON serializable)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")

    # Define the payload
    payload = {
        "data": data.to_dict(orient="records"),
    }

    # Define the query parameters
    params = {"symbol": symbol, "model_type": model_type, "phase": phase}

    async with httpx.AsyncClient(timeout=None) as client:
        # Send POST request to FastAPI endpoint
        response = await client.post(url, params=params, json=payload)

        # Check if the response is successful
        response.raise_for_status()

        result = response.json()

    if phase == "training":
        return (
            to_processed_data(result["data"]["train"]),
            to_processed_data(result["data"]["test"]),
        )
    else:
        return to_processed_data(result["data"])
