from prefect import flow
from .tasks import load_data, preprocess_data


@flow(name="Data Pipeline")
def run_data_pipeline(
    model_type: str, symbol: str, phase: str, data_service, processing_service
):
    raw_data_future = load_data.submit(data_service, symbol)
    raw_data = raw_data_future.result()

    preprocessed_data_future = preprocess_data.submit(
        service=processing_service,
        symbol=symbol,
        data=raw_data,
        model_type=model_type,
        phase=phase,
    )
    preprocessed_data = preprocessed_data_future.result()

    return preprocessed_data
