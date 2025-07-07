from prefect import task
import pandas as pd

from services.data_processing import DataProcessingService
from core.types import ProcessedData


@task(retries=2, retry_delay_seconds=5)
async def preprocess_data(
    service: DataProcessingService,
    symbol: str,
    data: pd.DataFrame,
    model_type: str,
    phase: str,
) -> ProcessedData:
    return await service.preprocess_data(
        symbol=symbol, data=data, model_type=model_type, phase=phase
    )


@task(retries=2, retry_delay_seconds=5)
async def postprocess_data(
    service: DataProcessingService,
    symbol: str,
    prediction,
    model_type: str,
    phase: str,
) -> ProcessedData:
    return await service.postprocess_data(
        symbol=symbol,
        prediction=prediction,
        model_type=model_type,
        phase=phase,
    )


@task(retries=2, retry_delay_seconds=5)
async def promote_scaler(service: DataProcessingService, model_type: str, symbol: str):
    await service.promote_scaler(model_type=model_type, symbol=symbol)
    return
