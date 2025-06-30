from prefect import flow
from .data_flow import run_data_pipeline
from .inference_flow import run_inference_pipeline
from .tasks import model_exist
from core.utils import get_model_name

PHASE = "prediction"


@flow(name="Prediction Pipeline", retries=2, retry_delay_seconds=10)
async def run_prediction_pipeline(
    model_type: str, symbol: str, data_service, processing_service, deployment_service
):
    live_model_exist = await model_exist(
        get_model_name(model_type, symbol, PHASE), deployment_service
    )
    if live_model_exist:
        prediction_input = await run_data_pipeline(
            symbol=symbol,
            model_type=model_type,
            data_service=data_service,
            processing_service=processing_service,
            phase=PHASE,
        )

        # Make prediction
        pred_target = await run_inference_pipeline(
            model_type=model_type,
            symbol=symbol,
            phase=PHASE,
            X=prediction_input.X,
            processing_service=processing_service,
            deployment_service=deployment_service,
        )

        return pred_target.y

    # No live model available
    return None
