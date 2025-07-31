from prefect import flow, task
from prefect.futures import wait
from itertools import product
from typing import Any
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
) -> dict[str, Any]:
    """
    Executes the prediction pipeline for a given model type and stock symbol.

    This task performs the following steps:
    1. Checks if a production model exists for the given model type and symbol.
    2. Loads recent stock data using the data service.
    3. Preprocesses the raw data to prepare it for prediction.
    4. Runs the inference flow to generate predictions and confidence scores.

    Args:
        model_type (str): The type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol

    Returns:
        dict: A dictionary containing the prediction results:
            - "y_pred": The predicted value(s)
            - "confidence": The confidence score(s) of the prediction(s)
            - "model_version": Version number of the model that made the prediction.
        Returns None if no production model is available.
    """

    # Generate the production model name
    production_model_name = get_model_name(model_type, symbol)

    # Check if it exist
    prod_model_exist = production_model_exists.submit(
        production_model_name,
    ).result()

    if prod_model_exist:
        # Load the recent stock data
        raw_data = load_recent_stock_data.submit(symbol=symbol)

        # Preprocess the raw data
        prediction_input = preprocess_data.submit(
            model_type=model_type,
            symbol=symbol,
            data=raw_data,
            phase=PHASE,
        )

        # Make prediction (prediction and confidence included)
        prediction_result = run_inference_flow(
            model_identifier=production_model_name,
            model_type=model_type,
            symbol=symbol,
            phase=PHASE,
            prediction_input=prediction_input,
        )

        return {
            "y_pred": prediction_result["prediction"].y,
            "confidence": prediction_result["confidence"],
            "model_version": prediction_result["model_version"],
        }

    # No live model available
    return None


@flow(
    name="Prediction Pipeline",
    description="Runs the full prediction pipeline (wrapper of the task `run_prediction_pipeline`)",
)
def run_prediction_flow(
    model_type: str,
    symbol: str,
) -> dict[str, Any]:
    """
    Run the prediction pipeline. It is basically a flow wrapper of the task in function
    `run_prediction_pipeline` (to make it a flow)

    Args:
        model_type (str): The type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol

    Returns:
        dict: A dictionary containing the prediction results:
            - "y_pred": The predicted value(s)
            - "confidence": The confidence score(s) of the prediction(s)
            - "model_version": Version number of the model that made the prediction.
        Returns None if no production model is available.
    """
    prediction_result = run_prediction_pipeline.submit(
        model_type=model_type,
        symbol=symbol,
    ).result()

    return prediction_result


@flow(
    name="Batch Predictions",
    description="Runs batch predictions across multiple model types and symbols combinations.",
)
def run_batch_prediction(
    model_types: list[str],
    symbols: list[str],
):
    """
    Executes batch predictions for all combinations of given model types and stock symbols.

    Args:
        model_types (list[str]): A list of model types to use (e.g., ["lstm", "xgboost"]).
        symbols (list[str]): A list of stock symbols to generate predictions for.
    """

    # Generate (model_type, symbol) pairs
    model_symbol_pairs = list(product(model_types, symbols))

    # Unpack the pairs into two lists for mapping
    model_type_list = [pair[0] for pair in model_symbol_pairs]
    symbol_list = [pair[1] for pair in model_symbol_pairs]

    # Run predictions using mapping
    predictions = run_prediction_pipeline.map(
        model_type=model_type_list,
        symbol=symbol_list,
    )

    # Wait for all predictions to complete
    wait(predictions)


@task(
    name="Historical Predictions Pipeline Task",
    description="Runs a prediction pipeline for a given symbol and date, using a deployed production model.",
)
def historical_prediction(
    model_type: str,
    symbol: str,
    end_date: datetime,
) -> dict[str, Any]:
    """
    Run a historical prediction for a given stock symbol and date.

    Args:
        model_type (str): The type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker sym
        end_date (datetime): The day before the day to predict

    Returns:
        dict: A dictionary containing the prediction results:
            - processed_y_pred (Any): Postprocessed prediction object.
            - confidence (float or None): Optional prediction confidence score.
            - model_version (int): Version number of the model that made the prediction.
    """

    # Load the raw historical data
    raw_data = load_historical_stock_prices_from_end_date.submit(
        symbol=symbol,
        end_date=end_date,
        days_back=config.preprocessing.SEQUENCE_LENGTH,
    )

    # Preprocess the raw data
    prediction_input = preprocess_data.submit(
        model_type=model_type,
        symbol=symbol,
        data=raw_data,
        phase=PHASE,
    )

    # Generate the production model name
    production_model_name = get_model_name(model_type, symbol)

    # Make prediction (prediction and confidence included)
    prediction_result = run_inference_flow(
        model_identifier=production_model_name,
        model_type=model_type,
        symbol=symbol,
        phase=PHASE,
        prediction_input=prediction_input,
    )

    return {
        "y_pred": prediction_result["prediction"].y,
        "confidence": prediction_result["confidence"],
        "model_version": prediction_result["model_version"],
    }


@flow(
    name="Historical Predictions Pipeline",
    description="Runs predictions for dates in a date range",
)
def run_historical_predictions_flow(
    model_type: str,
    symbol: str,
    trading_days: list[datetime],
):
    """
    Run historical predictions for a given symbol and date range using a production model.

    Args:
        model_type (str): The type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol
        trading_days (list[datetime]): The trading days we want to predict

    Returns:
        list|None: List of prediction results corresponding to those dates.
            Returns None if no production model is available.
    """

    # Shift all the dates by minus 1 (because we want all the sequence before the date to not see in the future)
    dates = [d - timedelta(days=1) for d in trading_days]

    # Generate the production model name
    production_model_name = get_model_name(model_type, symbol)

    # Check if it exist
    prod_model_exist = production_model_exists.submit(
        production_model_name,
    ).result()

    if prod_model_exist:

        # Make predictions
        prediction_futures = historical_prediction.map(
            model_type=model_type,
            symbol=symbol,
            end_date=dates,
        )

        # Wait for predictions and collect results
        prediction_results = [future.result() for future in prediction_futures]

        return prediction_results

    # There is no available production model
    return None
