from prefect import flow
from typing import Optional

from .prediction_flow import run_inference_flow
from ..tasks.data import load_recent_stock_data, postprocess_data, preprocess_data
from ..tasks.deployment import production_model_exists
from ..tasks.evaluation import evaluate, log_metrics_to_mlflow
from core.utils import get_model_name
from core.types import ProcessedData


@flow(
    name="Evaluation Pipeline",
    description="Evaluates the current production model performance using the most recent stock data.",
)
def run_evaluation_flow(
    model_type: str,
    symbol: str,
) -> Optional[dict[str, float]]:
    """
    Run evaluation on the current production model using recent stock data.

    This flow performs the following steps:
    1. Checks if a production model exists for the given symbol and model type.
    2. If a model exists:
        - Loads the most recent stock data using the data service.
        - Preprocesses the data for evaluation.
        - Runs model evaluation using the evaluation service.
    3. Returns evaluation metrics if a production model exists, else returns None.

    Parameters:
        model_type (str): The type of model (e.g. 'lstm', 'prophet').
        symbol (str): Stock ticker symbol.

    Returns:
        Optional[Dict[str, Any]]: Evaluation metrics if the production model exists, otherwise None.
    """
    # Get the production model name
    production_model_name = get_model_name(model_type, symbol)

    # Checks if the production model exists
    prod_model_exist = production_model_exists.submit(
        production_model_name,
    ).result()

    if prod_model_exist:

        # Load the recent stock data
        raw_data = load_recent_stock_data.submit(symbol=symbol).result()

        # Preprocess the raw data
        eval_data = preprocess_data.submit(
            model_type=model_type,
            symbol=symbol,
            data=raw_data,
            phase="evaluation",
        ).result()

        # Evaluate the training model
        metrics = evaluate_model(
            model_identifier=production_model_name,
            model_type=model_type,
            symbol=symbol,
            phase="prediction",
            eval_data=eval_data,
        )

        return metrics

    # No live model available
    return None


@flow(
    name="Evaluate and logs metrics (Sub-Pipeline)",
    description="Evaluates a model predictions and logs the resulting metrics to MLflow.",
)
def run_evaluate_and_log_flow(
    model_identifier: str,
    model_type: str,
    symbol: str,
    true_target: ProcessedData,
    pred_target: ProcessedData,
) -> dict[str, float]:
    """
    Evaluates the performance of a model predictions and logs the resulting metrics to MLflow.

    This sub-flow performs the following:
    1. Evaluates the model by comparing predicted and true target values using the evaluation service.
    2. Logs the resulting metrics to MLflow via the deployment service.

    Parameters:
        model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
        true_target (ProcessedData) : The true target values.
        pred_target (ProcessedData): The predicted target values.

    Returns:
        dict[str,float]: Dictionary of evaluation metrics (e.g., rmse, r2, etc).
    """

    # Evaluate the model
    metrics_future = evaluate.submit(
        true_target=true_target.y,
        pred_target=pred_target.y,
        model_type=model_type,
        symbol=symbol,
    )

    # Wait for it to finish and get result
    metrics = metrics_future.result()

    # Log the metrics
    future = log_metrics_to_mlflow.submit(
        model_identifier=model_identifier,
        metrics=metrics,
    )
    future.wait()

    return metrics


@flow(
    name="Evaluate model flow",
    description="Performs model inference, postprocesses the results, evaluates predictions, and logs metrics.",
)
def evaluate_model(
    model_identifier: str,
    model_type: str,
    symbol: str,
    phase: str,
    eval_data: ProcessedData,
) -> dict[str, float]:
    """
    Evaluates a model's predictions against true values and logs the resulting evaluation metrics.

    This flow performs the following steps:
    1. Runs the inference flow to generate model predictions.
    2. Postprocesses the true target values from the evaluation data.
    3. Evaluates the predicted vs. true values.
    4. Logs the evaluation metrics using MLflow.

    Parameters:
        model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
        model_type (str): The type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol.
        phase (str): The phase (e.g., "training", "evaluation", or "prediction").
        eval_data (ProcessedData): Preprocessed input data (for evaluation).

    Returns:
        dict[str,float]: Dictionary of evaluation metrics (e.g., rmse, r2, etc).
    """

    # Run inference (prediction) pipelines
    pred_target = run_inference_flow(
        model_identifier=model_identifier,
        model_type=model_type,
        symbol=symbol,
        phase=phase,
        prediction_input=eval_data,
    )["prediction"]

    # Postprocess ground truth
    true_target = postprocess_data.submit(
        symbol=symbol,
        model_type=model_type,
        phase=phase,
        prediction=eval_data.y,
    ).result()

    # Run evaluation and log metrics
    metrics = run_evaluate_and_log_flow(
        model_identifier=model_identifier,
        model_type=model_type,
        symbol=symbol,
        true_target=true_target,
        pred_target=pred_target,
    )
    return metrics
