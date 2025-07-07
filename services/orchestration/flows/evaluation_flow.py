from prefect import flow
from .data_flow import run_data_pipeline
from .inference_flow import run_inference_pipeline
from .evaluate_and_log_flow import run_evaluate_and_log_flow
from .tasks import postprocess_data, model_exist
from core.utils import get_model_name


@flow(name="Evaluation Pipeline")
def run_evaluation_pipeline(
    model_type: str,
    symbol: str,
    data_service,
    processing_service,
    deployment_service,
    evaluation_service,
):
    # Get the live model name
    live_model_name = get_model_name(model_type, symbol)

    # Checks if the live model exists
    live_model_exist = model_exist.submit(live_model_name, deployment_service)

    if live_model_exist.result():
        # Run the data pipeline (Data Ingestion + Preprocessing)
        eval_data = run_data_pipeline(
            symbol=symbol,
            model_type=model_type,
            data_service=data_service,
            processing_service=processing_service,
            phase="evaluation",
        )

        # Make inference (prediction) using live model
        pred_target, _, _ = run_inference_pipeline(
            model_identifier=live_model_name,
            model_type=model_type,
            symbol=symbol,
            phase="prediction",
            prediction_input=eval_data,
            processing_service=processing_service,
            deployment_service=deployment_service,
        )

        # Unnormalize (unscale) the ground truth values
        true_target_future = postprocess_data.submit(
            service=processing_service,
            symbol=symbol,
            model_type=model_type,
            phase="prediction",
            prediction=eval_data.y,
        )
        true_target = true_target_future.result()

        # Evaluate the training model
        metrics = run_evaluate_and_log_flow(
            model_identifier=live_model_name,
            true_target=true_target,
            pred_target=pred_target,
            evaluation_service=evaluation_service,
            deployment_service=deployment_service,
        )

        return metrics

    # No live model available
    return None
