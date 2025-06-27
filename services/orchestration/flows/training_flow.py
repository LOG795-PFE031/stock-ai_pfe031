from prefect import flow
from .tasks import train, evaluate
from .data_flow import run_data_pipeline
from core.utils import get_model_name

PHASE = "training"


@flow(name="Training Pipeline", retries=2, retry_delay_seconds=10)
async def run_training_pipeline(
    model_type,
    symbol,
    data_service,
    preprocessing_service,
    training_service,
    deployment_service,
):
    # Runt the data pipeline (Data Ingestion + Preprocessing)
    preprocessed_data = await run_data_pipeline(
        symbol=symbol,
        model_type=model_type,
        data_service=data_service,
        preprocessing_service=preprocessing_service,
        phase=PHASE,
    )

    # Split into train and test datasets
    training_data, test_data = preprocessed_data

    # Train the model
    training_results = await train(
        symbol=symbol,
        model_type=model_type,
        training_data=training_data,
        service=training_service,
    )

    # Get the train model name
    model_name = training_results["model_name"]

    # Evaluate the train model
    metrics = evaluate(
        model_name=model_name, X=test_data.X, y=test_data.y, service=deployment_service
    )

    # promote_if_better()

    # TODO What is the rest

    return metrics
