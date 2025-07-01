from prefect import flow
from .tasks import predict, postprocess_data, calculate_prediction_confidence
from core.utils import get_model_name
import numpy as np


@flow(name="Inference Pipeline")
async def run_inference_pipeline(
    model_type, symbol, phase, prediction_input, processing_service, deployment_service
):
    model_name = get_model_name(model_type=model_type, symbol=symbol, phase=phase)

    # Prediction
    y_pred = await predict(
        model_name=model_name,
        X=prediction_input.X,
        service=deployment_service,
    )

    processed_y_pred = await postprocess_data(
        service=processing_service,
        symbol=symbol,
        prediction=y_pred,
        model_type=model_type,
        phase=phase,
    )

    # Calculate prediction confidence (if the phase is prediction else None)
    confidence = (
        await calculate_prediction_confidence(
            model_type=model_type,
            y_pred=y_pred,
            prediction_input=prediction_input,
            service=deployment_service,
        )
        if phase == "prediction"
        else None
    )

    return processed_y_pred, confidence
