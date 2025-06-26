from prefect import task
import pandas as pd

from services.preprocessing import PreprocessingService
from core.types import FormattedInput


@task
async def preprocess_data(
    service: PreprocessingService,
    symbol: str,
    data: pd.DataFrame,
    model_type: str,
    phase: str,
) -> FormattedInput:
    return await service.get_preprocessed_data(
        symbol=symbol, data=data, model_type=model_type, phase=phase
    )
