import httpx
from prefect import task

from core.config import config


@task(
    name="promote_scaler",
    description="Promote a trained scaler to production",
    retries=2,
    retry_delay_seconds=5,
)
async def promote_scaler(model_type: str, symbol: str):
    """
    Promote a scaler to production using a data processing service.

    Args:
        model_type (str): Type of model (e.g. "prophet", "lstm").
        symbol (str): Stock ticker symbol.
    """

    # Get the endpoint URL
    url = (
        f"http://{config.data_processing_service.HOST}:"
        f"{config.data_processing_service.PORT}"
        "/processing/promote-scaler"
    )

    # Define the query parameters
    params = {
        "symbol": symbol,
        "model_type": model_type,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)

        # Check if the response is successful
        response.raise_for_status()
        return response.json()
