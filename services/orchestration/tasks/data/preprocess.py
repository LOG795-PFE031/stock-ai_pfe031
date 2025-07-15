from prefect import task
import pandas as pd
from typing import Union

from services.data_processing import DataProcessingService
from core.types import ProcessedData


@task(
    name="preprocess",
    description="Preprocess stock data for a given symbol and model type using a data processing service.",
    retries=2,
    retry_delay_seconds=5,
)
async def preprocess_data(
    service: DataProcessingService,
    symbol: str,
    data: pd.DataFrame,
    model_type: str,
    phase: str,
) -> Union[ProcessedData, tuple[ProcessedData, ProcessedData]]:
    """
    Prefect task to preprocess raw stock data for model consumption.

    Args:
        service (DataProcessingService): Service responsible for data preprocessing logic.
        symbol (str): The stock ticker symbol to preprocess data for.
        data (pd.DataFrame): Raw stock data to preprocess.
        model_type (str): The type of model (e.g., "lstm", "prophet")
        phase (str): Phase in the pipeline (e.g. "training", "prediction", "evaluation").

    Returns:
        ProcessedData: The preprocessed data
    """
    preprocessed_data = await service.preprocess_data(
        symbol=symbol, data=data, model_type=model_type, phase=phase
    )

    return preprocessed_data
