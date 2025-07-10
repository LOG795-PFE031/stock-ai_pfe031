from prefect import task
from typing import Any, Union
import numpy as np
import pandas as pd
from services import DeploymentService


@task(
    name="model_prediction",
    description="Make a prediction using a specified MLFlow model and input data.",
    retries=3,
    retry_delay_seconds=5,
)
async def predict(
    model_identifier: str,
    X: Union[pd.DataFrame, np.ndarray, list],
    service: DeploymentService,
) -> dict[Any, int]:
    """
    Make predictions using a MLFlow model and input data.

    Args:
        model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
        X: Input data for prediction.
        service (DeploymentService): Service that handles the model prediction.

    Returns:
        Prediction result.
    """
    return await service.predict(model_identifier, X)
