from prefect import task
from services.data_processing import DataProcessingService


@task(
    name="promote_scaler",
    description="Promote a trained scaler to production",
    retries=2,
    retry_delay_seconds=5,
)
async def promote_scaler(service: DataProcessingService, model_type: str, symbol: str):
    """
    Promote a scaler to production using a data processing service.

    Args:
        service (DataProcessingService): Data service handling scalers.
        model_type (str): Type of model (e.g. "prophet", "lstm").
        symbol (str): Stock ticker symbol.
    """
    await service.promote_scaler(model_type=model_type, symbol=symbol)
    return
