from prefect import task
import pandas as pd

from services.preprocessing import PreprocessingService, FormattedInput


@task
def preprocess_data(
    service: PreprocessingService,
    symbol: str,
    data: pd.DataFrame,
    model_type: str,
    phase: str,
) -> FormattedInput:
    return service.get_preprocessed_data(
        symbol=symbol, data=data, model_type=model_type, phase=phase
    )
