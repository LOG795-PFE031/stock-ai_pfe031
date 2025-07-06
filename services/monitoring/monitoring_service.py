import asyncio
from datetime import datetime, timedelta
from typing import Dict, Tuple

from services.base_service import BaseService
from core.logging import logger
from services.training.training_service import TrainingService
from services.model_service import ModelService
from services.data_service import DataService
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
        check_interval_seconds: int = 600,
    ):
        super().__init__()
        self.logger = logger["monitoring"]
        self.deployment_service = deployment_service
        self.orchestration_service = orchestration_service
        self.check_interval_seconds = check_interval_seconds
        self._last_mae: Dict[Tuple[str, str], float] = {}
        
        self.logger.info("Initializing MonitoringService...")
    
    async def initialize(self) -> None:
        """Initialize the Monitoring service."""
        self.logger.info("Initialized successfully")
        self._initialized = True
        
    async def cleanup(self) -> None:
        self.logger.info("Cleaned up successfully")
        await self.stop()

    # Check and trigger retraining
    async def run_once(self) -> None:
        """
        Perform a single drift-detection pass:
        1. List all live models.
        2. Read the latest MAE from Prometheus.
        3. If MAE increased since last check, trigger retraining.
        """
        try:
            self.logger.info("Starting monitoring drift check")
            live_models = await self.deployment_service.list_models()
            self.logger.debug(f"Found {len(live_models)} live models: {live_models}")

            for full_name in live_models:
                model_type, symbol, _ = full_name.split("_")
                symbol = symbol.upper()
                self.logger.debug(f"Checking model {model_type}_{symbol}")

                current_mae = evaluation_mae.labels(
                    symbol=symbol,
                    model_type=model_type
                )._value.get()
                if current_mae is None:
                    self.logger.debug(f"No MAE metric yet for {model_type}_{symbol}, skipping")
                    continue

                self.logger.debug(f"Current MAE for {model_type}_{symbol}: {current_mae:.4f}")
                key = (model_type, symbol)
                previous_mae = self._last_mae.get(key)
                if previous_mae is not None:
                    self.logger.debug(f"Previous MAE for {model_type}_{symbol}: {previous_mae:.4f}")
                self._last_mae[key] = current_mae

                if previous_mae is not None:
                    if current_mae > previous_mae:
                        self.logger.info(
                            f"MAE ↑ for {model_type}_{symbol}: {previous_mae} → {current_mae}"
                        )
                        self.logger.info(f"Triggering retraining for {model_type}_{symbol}")
                        try:
                            await self.orchestration_service.run_training_pipeline(model_type, symbol)
                            self.logger.info(f"Retraining triggered successfully for {model_type}_{symbol}")
                        except Exception as e:
                            self.logger.error(f"Failed to trigger retraining for {model_type}_{symbol}: {e}", exc_info=True)
                    else:
                        self.logger.info(f"No retraining needed for {model_type}_{symbol}: MAE did not increase ({previous_mae:.4f} → {current_mae:.4f})")
                else:
                    self.logger.debug(f"First MAE record for {model_type}_{symbol}: {current_mae:.4f}, no prior comparison")
        except Exception as e:
            self.logger.error(f"Error in monitoring run_once: {e}", exc_info=True)