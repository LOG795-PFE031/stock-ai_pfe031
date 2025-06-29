from prefect import flow
from .tasks import load_data, preprocess_data


@flow(name="Data Pipeline")
async def run_data_pipeline(
    model_type: str, symbol: str, phase: str, data_service, processing_service
):
    raw_data = await load_data(data_service, symbol)
    preprocessed_data = await preprocess_data(
        service=processing_service,
        symbol=symbol,
        data=raw_data,
        model_type=model_type,
        phase=phase,
    )

    return preprocessed_data
