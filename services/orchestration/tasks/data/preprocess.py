from prefect import task
import pandas as pd
import numpy as np
from typing import Any, Union, Dict, Tuple

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
# ) -> Union[ProcessedData, tuple[ProcessedData, ProcessedData]]:
) -> Union[dict[str, Any], Tuple[dict[str, Any], dict[str, Any]]]:
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
    # preprocessed_data = await service.preprocess_data(
    #     symbol=symbol, data=data, model_type=model_type, phase=phase
    # )

    # return preprocessed_data
    
    result = await service.preprocess_data(
        symbol=symbol, data=data, model_type=model_type, phase=phase
    )

    def to_dict(proc) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if proc.X is not None:
            if isinstance(proc.X, pd.DataFrame):
                # d["X"] = proc.X.to_dict(orient="records")
                d["X"] = proc.X.values.tolist()
            elif isinstance(proc.X, np.ndarray):
                d["X"] = proc.X.tolist()
            else:
                d["X"] = proc.X
        if getattr(proc, "y", None) is not None:
            d["y"] = proc.y.tolist() if isinstance(proc.y, np.ndarray) else proc.y
        if getattr(proc, "feature_index_map", None):
            d["feature_index_map"] = proc.feature_index_map
        if getattr(proc, "start_date", None):
            d["start_date"] = proc.start_date.isoformat()
        if getattr(proc, "end_date", None):
            d["end_date"] = proc.end_date.isoformat()
        return d

    if isinstance(result, tuple):
        train_proc, test_proc = result
        return to_dict(train_proc), to_dict(test_proc)
    else:
        return to_dict(result)
