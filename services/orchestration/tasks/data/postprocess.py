from prefect import task
import numpy as np
from typing import Any, Dict

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
# ) -> ProcessedData:
) -> dict[str, any]:
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
    # return await service.postprocess_data(
    #     symbol=symbol,
    #     prediction=prediction,
    #     model_type=model_type,
    #     phase=phase,
    # )

    proc = await service.postprocess_data(
        symbol=symbol,
        prediction=prediction,
        model_type=model_type,
        phase=phase,
    )
    
    # if it's already a dict, just return it
    if isinstance(proc, dict):
        return proc

    result: Dict[str, Any] = {}
    if proc.y is not None:
        result["y"] = proc.y.tolist() if isinstance(proc.y, np.ndarray) else proc.y
    if proc.X is not None:
        result["X"] = proc.X.tolist() if hasattr(proc.X, "tolist") else proc.X
    if proc.feature_index_map:
        result["feature_index_map"] = proc.feature_index_map
    if proc.start_date:
        result["start_date"] = proc.start_date.isoformat()
    if proc.end_date:
        result["end_date"] = proc.end_date.isoformat()
    return result

    return result