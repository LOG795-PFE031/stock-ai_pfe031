from prefect import task
from typing import Any
from services import TrainingService


@task(
    name="model_training",
    description="Train a model for a given symbol using training data.",
    retries=2,
    retry_delay_seconds=5,
)
async def train(
    symbol: str, model_type: str, training_data, service: TrainingService
) -> dict[str, Any]:
    """
    Train a model for a given symbol using training data.

    Args:
        symbol (str): Stock ticker symbol.
        model_type (str): Type of model (e.g. "prophet", "lstm").
        training_data: The data used for training the model.
        service (TrainingService): Service to perform the model training.

    Returns:
        dict[str,Any]: Training infos (containing the `run_id` key to locate the training model)
    """
    return await service.train_model(
        symbol=symbol, model_type=model_type, data=training_data
    )
