from prefect import flow
from .data_flow import run_data_pipeline


@flow(name="Prediction Pipeline", retries=2, retry_delay_seconds=10)
async def run_prediction_pipeline(
    model_type, symbol, data_service, preprocessing_service
):

    prediction_input = await run_data_pipeline(
        symbol=symbol,
        model_type=model_type,
        data_service=data_service,
        preprocessing_service=preprocessing_service,
        phase="prediction",
    )

    # predict(symbol=symbol, model_type=model_type, input=prediction_input)

    # TODO What is the rest
    return prediction_input
