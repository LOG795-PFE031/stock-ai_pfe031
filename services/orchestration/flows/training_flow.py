from prefect import flow
from .tasks import train, evaluate, postprocess_data
from .data_flow import run_data_pipeline
from .inference_flow import run_inference_pipeline

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

    model_name = training_results["model_name"]

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

    print("TRUEEEE TARGET : ", true_target)

    print("PREEEEED TARGET : ", pred_target)

    # Evaluate the training model
    metrics = await evaluate(
        model_name=model_name,
        true_target=true_target.y,
        pred_target=pred_target.y,
        service=evaluation_service,
    )

    return {
        "model_name": model_name,  # TODO modify with correct name (prediciton)
        "metrics": metrics,
        "training_history": training_results["training_history"],
        "run_info": training_results["run_info"],
    }
