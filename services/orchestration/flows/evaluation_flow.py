from prefect import flow
from .data_flow import run_data_pipeline
from .inference_flow import run_inference_pipeline
from .tasks import evaluate, postprocess_data

PHASE = "evaluation"


@flow(name="Evaluation Pipeline", retries=2, retry_delay_seconds=10)
async def run_evaluation_pipeline(
    model_type: str,
    symbol: str,
    data_service,
    processing_service,
    deployment_service,
):
    # Run the data pipeline (Data Ingestion + Preprocessing)
    eval_data = await run_data_pipeline(
        symbol=symbol,
        model_type=model_type,
        data_service=data_service,
        processing_service=processing_service,
        phase=PHASE,
    )

    # Make inference (prediction)
    y_pred = await run_inference_pipeline(
        model_type=model_type,
        symbol=symbol,
        phase=PHASE,
        X=eval_data.X,
        processing_service=processing_service,
        deployment_service=deployment_service,
    )

    # Unnormalize (unscale) the ground truth values
    y_true = await postprocess_data(
        service=processing_service,
        symbol=symbol,
        model_type=model_type,
        phase=PHASE,
        targets=eval_data.y,
    )

    """metrics = evaluate(
        symbol=symbol,
        model_type=model_type,
        Xtest=eval_data.X,
        ytest=eval_data.y,
        deployment_service=deployment_service
        production=True,
    )"""

    # return metrics
    return eval_data
