import asyncio
from services.deployment.deployment_service import DeploymentService
from services.orchestration.orchestration_service import OrchestrationService

class EvaluationSchedulerService:
    def __init__(
        self,
        deployment_service: DeploymentService,
        orchestration_service: OrchestrationService,
        interval_seconds: int = 300,
    ):
        self.deployment_service = deployment_service
        self.orchestration_service = orchestration_service
        self.interval_seconds = interval_seconds

    async def start(self):
        while True:
            models = await self.deployment_service.list_models()
            # run all evaluations in parallel
            tasks = []
            for full_name in models:
                model_type, symbol, _ = full_name.split("_")
                tasks.append(self.orchestration_service.run_evaluation_pipeline(model_type, symbol))
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.interval_seconds)
