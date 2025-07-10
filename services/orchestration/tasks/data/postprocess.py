from prefect import task

from services.data_processing import DataProcessingService
from core.types import ProcessedData


@task(
    name="postprocess",
    description="Postprocess a stock prediction for a given symbol and model type using a data processing service.",
    retries=2,
    retry_delay_seconds=5,
)
async def postprocess_data(
    service: DataProcessingService,
    symbol: str,
    prediction,
    model_type: str,
    phase: str,
) -> ProcessedData:
    """
    Postprocess stock prediction using a data processing service.

    Args:
        service (DataProcessingService): Data service for postprocessing.
        symbol (str): Stock ticker symbol.
        prediction: Predicted data.
        model_type (str): Type of model (e.g. "prophet", "lstm").
        phase (str): Phase in the pipeline. (e.g. "training", "prediction", "evaluation").

    Returns:
        ProcessedData: Postprocessed prediction result.
    """
    return await service.postprocess_data(
        symbol=symbol,
        prediction=prediction,
        model_type=model_type,
        phase=phase,
    )
