from prefect import flow
from prefect.logging import get_run_logger
from typing import Any

from core.utils import get_model_name
from services import (
    DataService,
    DeploymentService,
    EvaluationService,
)
from ..tasks.data import load_recent_stock_data, preprocess_data
from ..tasks.training import train
from ..tasks.deployment import production_model_exists
from .deployment_flow import run_deploy_flow
from .evaluation_flow import evaluate_model


PHASE = "training"


@flow(
    name="Training Pipeline",
    description="Train a machine learning model, evaluate it against a production model, and optionally deploy it.",
)
def run_training_flow(
    model_type: str,
    symbol: str,
    data_service: DataService,
    deployment_service: DeploymentService,
    evaluation_service: EvaluationService,
) -> dict[str, Any]:
    """
    Orchestrates the training, evaluation, and potential deployment of a training model
    for a specific stock symbol and model type.

    The pipeline performs the following steps:
    1. Loads the latest stock data.
    2. Preprocesses the data for training and evaluation.
    3. Trains a new model and evaluates both the new (candidate) and current production model.
    4. Compares performance metrics.
    5. Decides whether to promote the candidate model to production.

    Args:
        model_type (str): Type of model (e.g., "lstm", "prophet").
        symbol (str): Stock ticker symbol.
        data_service: Service used to load raw market data.
        deployment_service: Service used to perform predictions and manage models.
        evaluation_service: Service used to evaluate and compare models based on performance metrics.

    Returns:
        dict: Dictionary containing:
            - "training_results": Training metadata and run ID.
            - "metrics": Performance metrics for the trained (candidate) model.
            - "deployment_results": Outcome of the deployment step (success/failure).
    """

    # Get Prefect logger
    logger = get_run_logger()

    # Load the recent stock data
    raw_data = load_recent_stock_data.submit(data_service, symbol)

    # --- Training of the model ---

    # Preprocess the raw data
    preprocessed_data = preprocess_data.submit(
        model_type=model_type,
        symbol=symbol,
        data=raw_data,
        phase=PHASE,
    )

    # Split into train and test datasets
    training_data, test_data = preprocessed_data.result()

    # Train the model
    logger.info("Starting model training...")
    training_results_future = train.submit(
        symbol=symbol,
        model_type=model_type,
        training_data=training_data,
    )

    # --- Evaluation of the production model ---
    live_metrics = None

    # Get the phase and model name of production
    production_phase = "prediction"
    production_model_name = get_model_name(model_type=model_type, symbol=symbol)

    prod_model_exists = production_model_exists.submit(
        prod_model_name=production_model_name,
    ).result()

    if prod_model_exists:

        # Get the production evaluation data
        prod_eval_data = preprocess_data.submit(
            model_type=model_type,
            symbol=symbol,
            data=raw_data,
            phase="evaluation",
        )

        # Evaluate the production model
        live_metrics = evaluate_model(
            model_identifier=production_model_name,
            model_type=model_type,
            symbol=symbol,
            phase=production_phase,
            eval_data=prod_eval_data,
            deployment_service=deployment_service,
            evaluation_service=evaluation_service,
        )

    # --- Evaluation of the training model

    # Wait for the training results
    logger.info("Waiting for training to finish...")
    training_results = training_results_future.result()

    # Retrieve the run id
    run_id = training_results["run_id"]
    logger.info(f"Training completed. Run ID: {run_id}")

    # Evaluate the training model
    candidate_metrics = evaluate_model(
        run_id,
        model_type,
        symbol,
        PHASE,
        test_data,
        deployment_service,
        evaluation_service,
    )

    # Deploy (or not) the training model
    logger.info("Starting deployment check and promotion (if applicable)...")

    deployment_results = run_deploy_flow(
        model_type=model_type,
        symbol=symbol,
        run_id=run_id,
        candidate_metrics=candidate_metrics,
        prod_model_name=production_model_name,
        live_metrics=live_metrics,
        evaluation_service=evaluation_service,
        deployment_service=deployment_service,
    )
    logger.info(f"Deployment process completed. Result: {deployment_results}")

    return {
        "training_results": training_results,
        "metrics": candidate_metrics,
        "deployment_results": deployment_results,
    }
