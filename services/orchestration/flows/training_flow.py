from prefect import flow
from .tasks import (
    train,
    postprocess_data,
    model_exist,
    should_deploy_model,
    promote_model,
    promote_scaler,
)
from .data_flow import run_data_pipeline
from .inference_flow import run_inference_pipeline
from core.utils import get_model_name
from .evaluate_and_log_flow import run_evaluate_and_log_flow

PHASE = "training"


@flow(name="Training Pipeline", retries=2, retry_delay_seconds=10)
async def run_training_pipeline(
    model_type,
    symbol,
    data_service,
    processing_service,
    training_service,
    deployment_service,
    evaluation_service,
):
    # Run the data pipeline (Data Ingestion + Preprocessing)
    preprocessed_data = await run_data_pipeline(
        symbol=symbol,
        model_type=model_type,
        data_service=data_service,
        processing_service=processing_service,
        phase=PHASE,
    )

    # Split into train and test datasets
    training_data, test_data = preprocessed_data

    # Train the model
    training_results = await train(
        symbol=symbol,
        model_type=model_type,
        training_data=training_data,
        service=training_service,
    )

    # Make inference (prediction)
    pred_target = await run_inference_pipeline(
        model_type=model_type,
        symbol=symbol,
        phase=PHASE,
        X=test_data.X,
        processing_service=processing_service,
        deployment_service=deployment_service,
    )

    # Postprocess the ground truth values
    true_target = await postprocess_data(
        service=processing_service,
        symbol=symbol,
        model_type=model_type,
        phase=PHASE,
        targets=test_data.y,
    )

    # Get the training model name
    model_name = get_model_name(model_type=model_type, symbol=symbol, phase=PHASE)

    # Evaluate the training model
    metrics = await run_evaluate_and_log_flow(
        model_name=model_name,
        true_target=true_target,
        pred_target=pred_target,
        evaluation_service=evaluation_service,
        deployment_service=deployment_service,
    )

    # Deploy the model
    deployment_results = await run_deployment_pipeline(
        model_type=model_type,
        symbol=symbol,
        candidate_metrics=metrics,
        data_service=data_service,
        processing_service=processing_service,
        deployment_service=deployment_service,
        evaluation_service=evaluation_service,
    )

    return {
        "training_results": training_results,
        "metrics": metrics,
        "deployment_results": deployment_results,
    }


@flow(name="Deployment Pipeline", retries=2, retry_delay_seconds=5)
async def run_deployment_pipeline(
    model_type: str,
    symbol: str,
    candidate_metrics,
    data_service,
    processing_service,
    deployment_service,
    evaluation_service,
):
    # This is the phase of the live model
    live_phase = "prediction"

    # Get the live model name
    live_model_name = get_model_name(
        model_type=model_type, symbol=symbol, phase=live_phase
    )

    deployment_results = None

    if await model_exist(live_model_name, deployment_service):

        # Run the data pipeline (Data Ingestion + Preprocessing)
        # This is done because the live model its data preprocess (own scaler)
        test_data = await run_data_pipeline(
            symbol=symbol,
            model_type=model_type,
            data_service=data_service,
            processing_service=processing_service,
            phase="evaluation",
        )

        # Make inference on the test data with the live model
        pred_target = await run_inference_pipeline(
            model_type=model_type,
            symbol=symbol,
            phase=live_phase,
            X=test_data.X,
            processing_service=processing_service,
            deployment_service=deployment_service,
        )

        # Postprocess the ground truth values
        true_target = await postprocess_data(
            service=processing_service,
            symbol=symbol,
            model_type=model_type,
            phase=live_phase,
            targets=test_data.y,
        )

        # Evaluate the live model
        live_metrics = await run_evaluate_and_log_flow(
            model_name=live_model_name,
            true_target=true_target,
            pred_target=pred_target,
            evaluation_service=evaluation_service,
            deployment_service=deployment_service,
        )

        # Promote the training model if better
        if await should_deploy_model(
            candidate_metrics=candidate_metrics,
            live_metrics=live_metrics,
            service=evaluation_service,
        ):
            deployment_results = promote_model(
                model_type=model_type, symbol=symbol, service=deployment_service
            )

    else:
        # Automatically promote training model (if there is no live model)
        deployment_results = promote_model(
            model_type=model_type, symbol=symbol, service=deployment_service
        )

    if deployment_results is not None:
        # If there was a deployment
        await promote_scaler(
            model_type=model_type, symbol=symbol, service=processing_service
        )

    return deployment_results
