from prefect import task
from services import TrainingService


@task
def train(service: TrainingService, symbol: str, model_type: str, training_data):
    return service.train_model(symbol=symbol, model_type=model_type, data=training_data)
