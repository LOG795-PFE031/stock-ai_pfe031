from prefect import task

from services import EvaluationService


@task(
    name="model_evaluation",
    description="Evaluate model predictions using the evaluation service.",
    retries=2,
    retry_delay_seconds=5,
)
async def evaluate(
    true_target, pred_target, service: EvaluationService
) -> dict[str, float]:
    """
    Evaluate model predictions using ground truth values.

    Args:
        true_target: Ground truth target values.
        pred_target: Predicted values from the model.
        service (EvaluationService): Evaluation service.

    Returns:
        dict[str,float]: Dictionary of evaluation metrics (e.g., rmse, r2, etc).
    """
    return await service.evaluate(y_true=true_target, y_pred=pred_target)
