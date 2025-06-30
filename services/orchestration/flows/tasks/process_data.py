from prefect import task
import pandas as pd

from services.data_processing import DataProcessingService
from core.types import PreprocessedData


@task
async def preprocess_data(
    service: DataProcessingService,
    symbol: str,
    data: pd.DataFrame,
    model_type: str,
    phase: str,
) -> PreprocessedData:
    return await service.preprocess_data(
        symbol=symbol, data=data, model_type=model_type, phase=phase
    )


@task
async def postprocess_data(
    service: DataProcessingService,
    symbol: str,
    targets,
    model_type: str,
    phase: str,
) -> PreprocessedData:
    return await service.postprocess_data(
        symbol=symbol, targets=targets, model_type=model_type, phase=phase
    )


@task
async def promote_scaler(service: DataProcessingService, model_type: str, symbol: str):
    await service.promote_scaler(model_type=model_type, symbol=symbol)
    return
