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


@flow(name="Training Pipeline")
def run_training_pipeline(
    model_type,
    symbol,
    data_service,
    processing_service,
    training_service,
    deployment_service,
    evaluation_service,
):
    # Run the data pipeline (Data Ingestion + Preprocessing)
    preprocessed_data = run_data_pipeline(
        symbol=symbol,
        model_type=model_type,
        data_service=data_service,
        processing_service=processing_service,
        phase=PHASE,
    )

    # Split into train and test datasets
    training_data, test_data = preprocessed_data

    # Train the model
    training_results_future = train.submit(
        symbol=symbol,
        model_type=model_type,
        training_data=training_data,
        service=training_service,
    )

    # Get the training run id
    training_results = training_results_future.result()
    run_id = training_results["run_id"]

    # Make inference (prediction)
    pred_target, _, _ = run_inference_pipeline(
        model_identifier=run_id,
        model_type=model_type,
        symbol=symbol,
        phase=PHASE,
        prediction_input=test_data,
        processing_service=processing_service,
        deployment_service=deployment_service,
    )

    # Postprocess the ground truth values
    true_target_future = postprocess_data.submit(
        service=processing_service,
        symbol=symbol,
        model_type=model_type,
        phase=PHASE,
        prediction=test_data.y,
    )
    true_target = true_target_future.result()

    # Evaluate the training model
    metrics = run_evaluate_and_log_flow(
        model_identifier=run_id,
        true_target=true_target,
        pred_target=pred_target,
        evaluation_service=evaluation_service,
        deployment_service=deployment_service,
    )

    # Deploy the model
    deployment_results = run_deployment_pipeline(
        run_id=run_id,
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


@flow(name="Deployment Pipeline")
def run_deployment_pipeline(
    run_id: str,
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
    live_model_name = get_model_name(model_type=model_type, symbol=symbol)

    deployment_results = None

    # Checks if the live model exists
    live_model_exist = model_exist.submit(live_model_name, deployment_service)

    if live_model_exist.result():

        # Run the data pipeline (Data Ingestion + Preprocessing)
        # This is done because the live model its data preprocess (own scaler)
        test_data = run_data_pipeline(
            symbol=symbol,
            model_type=model_type,
            data_service=data_service,
            processing_service=processing_service,
            phase="evaluation",
        )

        # Make inference on the test data with the live model
        pred_target, _, _ = run_inference_pipeline(
            model_identifier=live_model_name,
            model_type=model_type,
            symbol=symbol,
            phase=live_phase,
            prediction_input=test_data,
            processing_service=processing_service,
            deployment_service=deployment_service,
        )

        # Postprocess the ground truth values
        true_target_future = postprocess_data.submit(
            service=processing_service,
            symbol=symbol,
            model_type=model_type,
            phase=live_phase,
            prediction=test_data.y,
        )
        true_target = true_target_future.result()

        # Evaluate the live model
        live_metrics = run_evaluate_and_log_flow(
            model_identifier=live_model_name,
            true_target=true_target,
            pred_target=pred_target,
            evaluation_service=evaluation_service,
            deployment_service=deployment_service,
        )

        # Promote the training model if better
        should_deploy_train_model = should_deploy_model.submit(
            candidate_metrics=candidate_metrics,
            live_metrics=live_metrics,
            service=evaluation_service,
        )

        if should_deploy_train_model.result():
            deployment_results = promote_model(
                run_id=run_id, model_name=live_model_name, service=deployment_service
            )

    else:
        # Automatically promote training model (if there is no live model)
        deployment_results = promote_model(
            run_id=run_id, model_name=live_model_name, service=deployment_service
        )

    if deployment_results is not None:
        # If there was a deployment
        promote_scaler.submit(
            model_type=model_type, symbol=symbol, service=processing_service
        )

    return deployment_results
