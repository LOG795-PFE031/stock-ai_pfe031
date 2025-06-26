from prefect import task
from services import TrainingService


@task
async def train(symbol: str, model_type: str, training_data, service: TrainingService):
    return await service.train_model(
        symbol=symbol, model_type=model_type, data=training_data
    )
