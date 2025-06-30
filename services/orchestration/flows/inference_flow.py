from prefect import flow
from .tasks import predict, postprocess_data
from core.utils import get_model_name


@flow(name="Inference Pipeline")
async def run_inference_pipeline(
    model_type, symbol, phase, X, processing_service, deployment_service
):
    model_name = get_model_name(model_type=model_type, symbol=symbol, phase=phase)

    y_pred = await predict(model_name=model_name, X=X, service=deployment_service)

    processed_y_pred = await postprocess_data(
        service=processing_service,
        symbol=symbol,
        targets=y_pred,
        model_type=model_type,
        phase=phase,
    )

    return processed_y_pred
