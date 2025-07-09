import asyncio
import contextlib
from datetime import datetime
from typing import Dict, Tuple

from services.base_service import BaseService
from core.logging import logger
from services.orchestration.orchestration_service import OrchestrationService
from services.deployment.deployment_service import DeploymentService
from monitoring.prometheus_metrics import (
    prediction_time_seconds,
    prediction_confidence,
    evaluation_mae
)

class MonitoringService(BaseService):
    def __init__(
        self,
        deployment_service: DeploymentService,
        orchestration_service: OrchestrationService,
        check_interval_seconds: int = 24 * 60 * 60,  # default: once per day
    ):
        super().__init__()
        self.logger = logger["monitoring"]
        self.deployment_service = deployment_service
        self.orchestration_service = orchestration_service
        self.check_interval_seconds = check_interval_seconds
        self._last_mae: Dict[Tuple[str, str], float] = {}
        self._task: asyncio.Task | None = None
        self.logger.info("Initializing MonitoringService (daily checks)...")

    async def initialize(self) -> None:
        self._initialized = True
        # start background loop
        self._task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("MonitoringService initialized and loop started")

    async def cleanup(self) -> None:
        self._initialized = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.logger.info("MonitoringService cleaned up")

    async def _monitoring_loop(self) -> None:
        """Run run_once(), then sleep check_interval, forever."""
        try:
            while self._initialized:
                await self.run_once()
                await asyncio.sleep(self.check_interval_seconds)
        except asyncio.CancelledError:
            pass

    async def run_once(self) -> None:
        """
        1. List all live models (e.g. 'lstm_AAPL')
        2. For each: run evaluation, read MAE, compare to last run, retrain if worse.
        """
        self.logger.info("🛠️  Starting daily drift check")
        try:
            live_models = await self.deployment_service.list_models()
            for name in live_models:
                # split only on first underscore
                model_type, symbol = name.split("_", 1)
                symbol = symbol.upper()

                # 1) Run evaluation pipeline (updates the evaluation_mae gauge)
                result = await self.orchestration_service.run_evaluation_pipeline(
                    model_type, symbol
                )
                
                # self.logger.info(f"Evaluation pipeline result is: {result}")

                # 2) Grab the just-set MAE
                current = evaluation_mae.labels(
                    model_type=model_type, symbol=symbol
                )._value.get()
                if current is None:
                    self.logger.warning(f"No MAE for {name}, skipping drift check")
                    continue

                key = (model_type, symbol)
                prev = self._last_mae.get(key)
                self._last_mae[key] = current

                # 3) Compare and retrain if MAE worsened
                if prev is not None and current > prev:
                    self.logger.info(
                        f"MAE increased for {name}: {prev:.4f} → {current:.4f}; retraining..."
                    )
                    await self.orchestration_service.run_training_pipeline(
                        model_type, symbol
                    )
                else:
                    self.logger.info(f"No retraining needed for {name}, prev MAE: {prev} vs current MAE: {current}")
        except Exception as e:
            self.logger.error(f"Error in daily drift check: {e}")
        finally:
            self.logger.info("🏁 Daily drift check complete")