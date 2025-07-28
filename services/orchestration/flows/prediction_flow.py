from prefect import flow, task
from prefect.futures import wait
from itertools import product
from typing import Any, Callable
from datetime import datetime, timedelta

from ..tasks.prediction import predict, calculate_prediction_confidence
from ..tasks.data import (
    load_recent_stock_data,
    postprocess_data,
    preprocess_data,
    load_historical_stock_prices_from_end_date,
)
from ..tasks.deployment import production_model_exists
from core.utils import get_model_name
from services import DeploymentService
from core.types import ProcessedData
from core.config import config

PHASE = "prediction"


@flow(
    name="Inference Pipeline",
    description="Pipeline that runs inference on preprocessed stock data using "
    + "a deployed model, followed by optional confidence calculation and postprocessing.",
)
def run_inference_flow(
    model_identifier: str,
    model_type: str,
    symbol: str,
    phase: str,
    prediction_input: ProcessedData,
    deployment_service: DeploymentService,
) -> dict[str, Any]:
    """
    Run the full inference pipeline for a given model and stock symbol.

    This flow performs the following steps:
      1. Sends input features to a deployed model to get predictions.
      2. Optionally calculates confidence (if the phase is 'prediction' which is the 'production' phase).
      3. Postprocesses the raw predictions into final output format.

    Args:
        model_identifier (str): Identifier for the model (run ID of a
                logged model (training model) or name of a registered model (live model)).
        model_type (str): Type of the model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol
        phase (str): Phase of the pipeline (e.g., "prediction", "evaluation", 'training).
        prediction_input (ProcessedData): Preprocessed input features for prediction.
        deployment_service: Service used for predicting and calculating confidence.

    Returns:
        dict: A dictionary containing the prediction results:
            - processed_y_pred (Any): Postprocessed prediction object.
            - confidence (float or None): Optional prediction confidence score.
            - model_version (int): Version number of the model that made the prediction.
    """
    # Prediction
    predict_result = predict.submit(
        model_identifier=model_identifier,
        X=prediction_input.X,
        service=deployment_service,
    ).result()

    y_pred, model_version = (
        predict_result["predictions"],
        predict_result["model_version"],
    )

    # Postprocess the predictions
    processed_y_pred_future = postprocess_data.submit(
        symbol=symbol,
        prediction=y_pred,
        model_type=model_type,
        phase=phase,
    )

    # Calculate prediction confidence (if the phase is prediction else None)
    confidence = (
        calculate_prediction_confidence.submit(
            model_type=model_type,
            symbol=symbol,
            y_pred=y_pred,
            prediction_input=prediction_input,
            service=deployment_service,
        )
        if phase == "prediction"
        else None
    )

    # Wait for results
    processed_y_pred = processed_y_pred_future.result()
    if confidence:
        confidence = confidence.result()

    return {
        "prediction": processed_y_pred,
        "confidence": confidence,
        "model_version": model_version,
    }


@task(
    name="Prediction Pipeline Task",
    description="Runs the full prediction process for a given model and symbol, "
    + "including data loading, preprocessing, and inference using the latest production model.",
)
def run_prediction_pipeline(
    model_type: str,
    symbol: str,
    fetch_stock_data: Callable,
    deployment_service: DeploymentService,
) -> dict[str, Any]:
    """
    Run the full prediction pipeline for a given model and symbol.

    This task performs the following steps:
      1. Loads recent stock data for the symbol.
      2. Preprocesses the data for prediction.
      3. Runs inference using the latest production model.
      4. Returns the prediction results.

    Args:
        model_type (str): Type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol.
        fetch_stock_data (Callable): Function to fetch stock data from API.
        deployment_service: Service used for predicting and managing models.

    Returns:
        dict: A dictionary containing the prediction results:
            - prediction (Any): The predicted value.
            - confidence (float or None): Optional prediction confidence score.
            - model_version (int): Version number of the model that made the prediction.
    """
    # Load the recent stock data
    raw_data = load_recent_stock_data.submit(fetch_stock_data, symbol)

    # Preprocess the raw data
    preprocessed_data = preprocess_data.submit(
        model_type=model_type,
        symbol=symbol,
        data=raw_data,
        phase=PHASE,
    )

    # Get the production model name
    production_model_name = get_model_name(model_type=model_type, symbol=symbol)

    # Check if the production model exists
    prod_model_exists = production_model_exists.submit(
        prod_model_name=production_model_name, service=deployment_service
    ).result()

    if not prod_model_exists:
        return {
            "status": "error",
            "error": f"No production model available for {model_type}_{symbol}",
        }

    # Run the inference pipeline
    inference_result = run_inference_flow.submit(
        model_identifier=production_model_name,
        model_type=model_type,
        symbol=symbol,
        phase=PHASE,
        prediction_input=preprocessed_data,
        deployment_service=deployment_service,
    )

    return {
        "status": "success",
        "prediction": inference_result.result()["prediction"],
        "confidence": inference_result.result()["confidence"],
        "model_version": inference_result.result()["model_version"],
    }


@flow(
    name="Prediction Pipeline",
    description="Runs the full prediction pipeline (wrapper of the task `run_prediction_pipeline`)",
)
def run_prediction_flow(
    model_type: str,
    symbol: str,
    fetch_stock_data: Callable,
    deployment_service: DeploymentService,
) -> dict[str, Any]:
    """
    Run the full prediction pipeline for a given model and symbol.

    Args:
        model_type (str): Type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol.
        fetch_stock_data (Callable): Function to fetch stock data from API.
        deployment_service: Service used for predicting and managing models.

    Returns:
        dict: A dictionary containing the prediction results.
    """
    return run_prediction_pipeline.submit(
        model_type=model_type,
        symbol=symbol,
        fetch_stock_data=fetch_stock_data,
        deployment_service=deployment_service,
    ).result()


@flow(
    name="Batch Predictions",
    description="Runs batch predictions across multiple model types and symbols combinations.",
)
def run_batch_prediction(
    model_types: list[str],
    symbols: list[str],
    fetch_stock_data: Callable,
    deployment_service: DeploymentService,
):
    """
    Run batch predictions for multiple model types and symbols.

    Args:
        model_types (list[str]): List of model types to use for predictions.
        symbols (list[str]): List of stock symbols to predict for.
        fetch_stock_data (Callable): Function to fetch stock data from API.
        deployment_service: Service used for predicting and managing models.
    """
    # Create all combinations of model types and symbols
    combinations = list(product(model_types, symbols))

    # Submit all prediction tasks
    futures = [
        run_prediction_pipeline.submit(
            model_type=model_type,
            symbol=symbol,
            fetch_stock_data=fetch_stock_data,
            deployment_service=deployment_service,
        )
        for model_type, symbol in combinations
    ]

    # Wait for all tasks to complete
    wait(futures)

    # Collect results
    results = []
    for future in futures:
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return results


@task(
    name="Historical Predictions Pipeline Task",
    description="Runs a prediction pipeline for a given symbol and date, using a deployed production model.",
)
def historical_prediction(
    model_type: str,
    symbol: str,
    end_date: datetime,
    fetch_stock_data: Callable,
    deployment_service: DeploymentService,
) -> dict[str, Any]:
    """
    Run a prediction pipeline for a given symbol and date.

    Args:
        model_type (str): Type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol.
        end_date (datetime): End date for the historical data.
        fetch_stock_data (Callable): Function to fetch stock data from API.
        deployment_service: Service used for predicting and managing models.

    Returns:
        dict: A dictionary containing the prediction results.
    """
    # Load historical stock data
    raw_data = load_historical_stock_prices_from_end_date.submit(
        fetch_stock_data, symbol, end_date, 60  # 60 days back
    )

    # Preprocess the raw data
    preprocessed_data = preprocess_data.submit(
        model_type=model_type,
        symbol=symbol,
        data=raw_data,
        phase=PHASE,
    )

    # Get the production model name
    production_model_name = get_model_name(model_type=model_type, symbol=symbol)

    # Check if the production model exists
    prod_model_exists = production_model_exists.submit(
        prod_model_name=production_model_name, service=deployment_service
    ).result()

    if not prod_model_exists:
        return {
            "status": "error",
            "error": f"No production model available for {model_type}_{symbol}",
        }

    # Run the inference pipeline
    inference_result = run_inference_flow.submit(
        model_identifier=production_model_name,
        model_type=model_type,
        symbol=symbol,
        phase=PHASE,
        prediction_input=preprocessed_data,
        deployment_service=deployment_service,
    )

    return {
        "status": "success",
        "prediction": inference_result.result()["prediction"],
        "confidence": inference_result.result()["confidence"],
        "model_version": inference_result.result()["model_version"],
        "date": end_date,
    }


@flow(
    name="Historical Predictions Pipeline",
    description="Runs predictions for dates in a date range",
)
def run_historical_predictions_flow(
    model_type: str,
    symbol: str,
    trading_days: list[datetime],
    fetch_stock_data: Callable,
    deployment_service: DeploymentService,
):
    """
    Run predictions for a range of trading days.

    Args:
        model_type (str): Type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol.
        trading_days (list[datetime]): List of trading days to predict for.
        fetch_stock_data (Callable): Function to fetch stock data from API.
        deployment_service: Service used for predicting and managing models.

    Returns:
        dict: A dictionary containing the prediction results for all dates.
    """
    # Submit all historical prediction tasks
    futures = [
        historical_prediction.submit(
            model_type=model_type,
            symbol=symbol,
            end_date=date,
            fetch_stock_data=fetch_stock_data,
            deployment_service=deployment_service,
        )
        for date in trading_days
    ]

    # Wait for all tasks to complete
    wait(futures)

    # Collect results
    results = []
    for future in futures:
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return {
        "status": "success",
        "predictions": results,
    }
