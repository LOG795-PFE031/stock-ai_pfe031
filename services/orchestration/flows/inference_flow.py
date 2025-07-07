from prefect import flow
from .tasks import predict, postprocess_data, calculate_prediction_confidence
import numpy as np


@flow(name="Inference Pipeline")
def run_inference_pipeline(
    model_identifier,
    model_type,
    symbol,
    phase,
    prediction_input,
    processing_service,
    deployment_service,
):

    # Prediction
    predict_future = predict.submit(
        model_identifier=model_identifier,
        X=prediction_input.X,
        service=deployment_service,
    )
    y_pred, model_version = predict_future.result()

    processed_y_pred_future = postprocess_data.submit(
        service=processing_service,
        symbol=symbol,
        prediction=y_pred,
        model_type=model_type,
        phase=phase,
    )

    # Calculate prediction confidence (if the phase is prediction else None)
    confidence = (
        calculate_prediction_confidence.submit(
            model_type=model_type,
            y_pred=y_pred,
            prediction_input=prediction_input,
            service=deployment_service,
        ).result()
        if phase == "prediction"
        else None
    )

    # Wait for results
    processed_y_pred = processed_y_pred_future.result()

    return processed_y_pred, confidence, model_version
