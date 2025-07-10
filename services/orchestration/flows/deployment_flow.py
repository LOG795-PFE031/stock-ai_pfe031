from prefect import flow
from prefect.logging import get_run_logger
from typing import Any

from ..tasks.deployment import (
    is_ready_for_deployment,
    promote_model,
    promote_scaler,
)
from services import DataProcessingService, EvaluationService, DeploymentService


@flow(
    name="Deployment sub pipeline",
    description="Deploy a candidate model if it outperforms the production model, or if no production model exists.",
)
def run_deploy_flow(
    model_type: str,
    symbol: str,
    run_id: str,
    candidate_metrics: dict,
    prod_model_name: str | None,
    live_metrics: dict | None,
    evaluation_service: EvaluationService,
    deployment_service: DeploymentService,
    processing_service: DataProcessingService,
) -> dict[str, Any]:
    """
    Sub-flow responsible for deploying a trained model to production if it's better
    than the current production model (or if no production model exists).
    Also promotes the associated scaler after deployment.

    Args:
        model_type (str): Type of the model (e.g. "lstm", "prophet").
        symbol (str): Stock ticker symbol
        run_id (str): MLflow run ID of the trained candidate model.
        candidate_metrics (dict): Evaluation metrics of the newly trained model.
        prod_model_name (str): Name of the current production model (if any).
        live_metrics (dict): Evaluation metrics of the current production model.
        evaluation_service (EvaluationService): Service for comparing model performance.
        deployment_service (DeploymentService): Service for model promotion.
        processing_service (DataProcessingService): Service for promoting the scaler.

    Returns:
        dict[str,Any]: Results of the deployment
    """

    # Get Prefect logger
    logger = get_run_logger()

    # Initialize the deployment results
    deployment_results = None

    # Check if there was a results for the live model
    prod_model_exist = prod_model_name and live_metrics

    if prod_model_exist:
        logger.info(f"Production model exists: {bool(prod_model_exist)}")

        # Promote the training model if better
        logger.info("Comparing candidate model with production model...")
        should_deploy_train_model = is_ready_for_deployment.submit(
            candidate_metrics=candidate_metrics,
            live_metrics=live_metrics,
            service=evaluation_service,
        )
        if should_deploy_train_model.result():
            logger.info("Candidate model outperformed production. Promoting...")
            deployment_results = promote_model.submit(
                run_id=run_id,
                prod_model_name=prod_model_name,
                service=deployment_service,
            ).result()
            logger.info("Candidate model promoted to production.")
        else:
            logger.info(
                "Candidate model did not outperform production. Skipping deployment."
            )
    else:
        # Automatically promote training model (if there is no live model)
        logger.info("No live model found. Auto-promoting candidate model.")
        deployment_results = promote_model.submit(
            run_id=run_id, prod_model_name=prod_model_name, service=deployment_service
        ).result()
        logger.info(
            "Candidate model promoted by default (no production model to compare)."
        )

    if deployment_results is not None:
        # If there was a deployment
        logger.info("Promoting scaler for the newly deployed model...")
        promote_scaler.submit(
            model_type=model_type, symbol=symbol, service=processing_service
        ).result()
        logger.info("Scaler promoted.")

    return deployment_results
