from prefect import flow
from .tasks import train
from .data_flow import run_data_pipeline


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
        phase="training",
    )

    # Split into train and test datasets
    training_data, test_data = preprocessed_data

    # Train the model
    training_results = train(
        symbol=symbol,
        model_type=model_type,
        training_data=training_data,
        service=training_service,
    )

    """metrics = evaluate(
        symbol=symbol,
        model_type=model_type,
        Xtest=test_data.X,
        ytest=test_data.y,
        deployment_service=deployment_service
        production=False,
    )

    promote_if_better()"""

    # TODO What is the rest

    return training_results
