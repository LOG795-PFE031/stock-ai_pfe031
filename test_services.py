"""
Module to test the different ML Pipelines :

- Training pipeline
= Prediction pipeline
- Evaluatopm pipeline

This module can test call to services too.
"""

# Initialize services
from services import (
    DataService,
    DataProcessingService,
    TrainingService,
    DeploymentService,
    EvaluationService,
)


from services.orchestration import OrchestrationService
from core.utils import get_model_name


# Create service instances in dependency order
data_service = DataService()
preprocessing_service = DataProcessingService()
training_service = TrainingService()
deployment_service = DeploymentService()
evaluation_service = EvaluationService()
orchestation_service = OrchestrationService(
    data_service=data_service,
    preprocessing_service=preprocessing_service,
    training_service=training_service,
    deployment_service=deployment_service,
    evaluation_service=evaluation_service,
)


async def main():
    # Initialize services in order of dependencies
    await data_service.initialize()  # No dependencies
    await preprocessing_service.initialize()
    await training_service.initialize()
    await deployment_service.initialize()
    await evaluation_service.initialize()
    await orchestation_service.initialize()

    model_type = "prophet"  # or "lstm"
    symbol = "AAPL"  # or "AMZN", etc.

    # Training pipeline
    # result = await orchestation_service.run_training_pipeline(model_type, symbol)

    # Prediction pipeline
    # result = await orchestation_service.run_prediction_pipeline(model_type, symbol)

    # Evaluation pipeline
    # result = await orchestation_service.run_evaluation_pipeline(model_type, symbol)

    # Test the deployment service model list models functions
    result = await deployment_service.list_models()

    # Print the results
    print(result)


# Run the main coroutine
import asyncio

asyncio.run(main())
