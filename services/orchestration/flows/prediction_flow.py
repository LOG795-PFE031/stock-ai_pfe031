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

    # Generate the production model (live model) name
    live_model_name = get_model_name(model_type, symbol)

    # Check if it exist
    live_model_exist = await model_exist(live_model_name, deployment_service)
    if live_model_exist:
        prediction_input = await run_data_pipeline(
            symbol=symbol,
            model_type=model_type,
            data_service=data_service,
            processing_service=processing_service,
            phase=PHASE,
        )

        # Make prediction (prediction and confidence included)
        pred_target, confidence, model_version = await run_inference_pipeline(
            model_identifier=live_model_name,
            model_type=model_type,
            symbol=symbol,
            phase=PHASE,
            prediction_input=prediction_input,
            processing_service=processing_service,
            deployment_service=deployment_service,
        )

        return {
            "y_pred": pred_target.y,
            "confidence": confidence,
            "model_version": model_version,
        }

    # No live model available
    return None
