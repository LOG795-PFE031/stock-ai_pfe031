import httpx
from prefect import task
import pandas as pd

from core.config import config
from core.types import ProcessedData
from ...utils import to_processed_data


@task(
    name="postprocess",
    description="Postprocess a stock prediction for a given symbol and model type using a data processing service.",
    retries=2,
    retry_delay_seconds=5,
)
async def postprocess_data(
    symbol: str,
    prediction,
    model_type: str,
    phase: str,
) -> ProcessedData:
    """
    Postprocess stock prediction using a data processing service.

    Args:
        symbol (str): Stock ticker symbol.
        prediction: Predicted data.
        model_type (str): Type of model (e.g. "prophet", "lstm").
        phase (str): Phase in the pipeline. (e.g. "training", "prediction", "evaluation").

    Returns:
        ProcessedData: Postprocessed prediction result.
    """

    # Get the endpoint URL
    url = (
        f"http://{config.data_processing_service.HOST}:"
        f"{config.data_processing_service.PORT}"
        "/processing/postprocess"
    )

    if hasattr(prediction, "tolist"):
        payload_data = prediction.tolist()
    elif hasattr(prediction, "to_dict"):
        # TODO : Déplacer cette logique dans le déploiement — la sortie du modèle Prophet
        # doit avoir les dates formatées en string pour être JSON-compatibles.
        for col in prediction.columns:
            if pd.api.types.is_datetime64_any_dtype(prediction[col]):
                prediction[col] = prediction[col].dt.strftime("%Y-%m-%d")
        payload_data = prediction.to_dict(orient="records")
    else:
        raise ValueError(
            "Unsupported prediction format. Must be a DataFrame or NumPy array."
        )

    # Define the payload
    payload = {"data": payload_data}

    # Define the query parameters
    params = {"symbol": symbol, "model_type": model_type, "phase": phase}

    async with httpx.AsyncClient(timeout=None) as client:
        # Send POST request to FastAPI endpoint
        response = await client.post(url, params=params, json=payload)

        # Check if the response is successful
        response.raise_for_status()

        result = response.json()

    return to_processed_data(result["data"])
