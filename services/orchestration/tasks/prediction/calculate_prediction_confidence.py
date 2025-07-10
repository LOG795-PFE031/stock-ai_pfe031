from prefect import task

from services import DeploymentService


@task(
    name="prediction_confidence_calculation",
    description="Calculate confidence scores for a model's predictions.",
    retries=3,
    retry_delay_seconds=5,
)
async def calculate_prediction_confidence(
    model_type: str, y_pred, prediction_input, service: DeploymentService
) -> list[float]:
    """
    Calculate prediction confidence scores using a deployment service.

    Args:
        model_type (str): Type of model (e.g. "prophet", "lstm").
        y_pred: Model output predictions.
        prediction_input: Original input data used for predictions.
        service (DeploymentService): Service handling deployment logic.

    Returns:
        list[float]: Confidence scores
    """
    return await service.calculate_prediction_confidence(
        model_type=model_type, prediction_input=prediction_input, y_pred=y_pred
    )
