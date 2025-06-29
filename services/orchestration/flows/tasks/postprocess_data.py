from prefect import task

from services.data_processing import DataProcessingService
from core.types import PreprocessedData


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
