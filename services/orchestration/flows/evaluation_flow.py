from prefect import flow
from .data_flow import run_data_pipeline


@flow(name="Evaluation Pipeline", retries=2, retry_delay_seconds=10)
async def run_evaluation_pipeline(
    model_type: str,
    symbol: str,
    data_service,
    preprocessing_service,
    deployment_service,
):

    eval_data = await run_data_pipeline(
        symbol=symbol,
        model_type=model_type,
        data_service=data_service,
        preprocessing_service=preprocessing_service,
        phase="evaluation",
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
