import asyncio
from datetime import datetime, timedelta

from prometheus_client import REGISTRY
from services.base_service import BaseService
from core.logging import logger
from services.training_service import TrainingService
from services.model_service import ModelService
from services.data_service import DataService
from monitoring.prometheus_metrics import (
    prediction_time_seconds,
    prediction_confidence,
    evaluation_mae
)

class MonitoringService(BaseService):
    def __init__(
        self,
        model_service: ModelService,
        data_service: DataService,
        training_service: TrainingService,
        check_interval_seconds: int = 600,
        latency_threshold: float = 3.0,
        confidence_threshold: float = 0.5,
        mae_threshold: float = 2.0,
        degradation_tolerance: int = 2,
    ):
        super().__init__()
        self.logger = logger["monitoring"]
        self.model_service = model_service
        self.data_service = data_service
        self.training_service = training_service
        self.check_interval = check_interval_seconds
        self.latency_threshold = latency_threshold
        self.confidence_threshold = confidence_threshold
        self.mae_threshold = mae_threshold
        self.degradation_tolerance = degradation_tolerance
        
        self._degraded_models = {}
        
        self.logger.info("Initializing MonitoringService...")
        
    async def initialize(self) -> None:
        """Initialize the Monitoring service."""
        self.logger.info("MONITORING service initialized successfully")
        self._initialized = True
        
    async def cleanup(self) -> None:
        self.logger.info("MonitoringService cleaned up successfully")
        await self.stop()

    # Check and trigger retraining
    async def check_all_models(self):
        self.logger.info("Monitoring, Checking all model metrics")
        models = await self.training_service.get_trainers()
        self.logger.info(f"Found {len(models)} models")
        for model in models:
            await self._check_model(model["symbol"], model["model_type"])

    async def _check_model(self, symbol: str, model_type: str):
        key = f"{symbol}_{model_type}"

        # Check MAE
        mae = self._get_gauge_value(evaluation_mae, model_type, symbol)

        if mae is not None and mae > self.mae_threshold:
            self.logger.warning(f"MAE drift detected for {key}: {mae:.4f}")
            self._degraded_models[key] = self._degraded_models.get(key, 0) + 1
            if self._degraded_models[key] >= self.degradation_tolerance:
                await self._trigger_retrain(symbol, model_type)
                self._degraded_models[key] = 0
            return


        # Check latency & confidence fallback
        latency = self._get_histogram_avg(prediction_time_seconds, model_type, symbol)
        confidence = self._get_gauge_value(prediction_confidence, model_type, symbol)

        if latency is None or confidence is None:
            return

        self.logger.info(f"[{key}] Latency: {latency:.2f}s | Confidence: {confidence:.2f}")

        if latency > self.latency_threshold or confidence < self.confidence_threshold:
            self._degraded_models[key] = self._degraded_models.get(key, 0) + 1
            self.logger.warning(f"Degradation detected for {key} (count={self._degraded_models[key]})")
            if self._degraded_models[key] >= self.degradation_tolerance:
                await self._trigger_retrain(symbol, model_type)
                self._degraded_models[key] = 0
        else:
            self._degraded_models[key] = 0

    def _get_histogram_avg(self, metric, model_type, symbol):
        try:
            samples = {
                s.name: s for s in metric.collect()[0].samples
                if s.labels.get("model_type") == model_type and s.labels.get("symbol") == symbol
            }
            count = samples.get(f"{metric._name}_count")
            sum_ = samples.get(f"{metric._name}_sum")
            if count and sum_ and count.value > 0:
                return float(sum_.value) / float(count.value)
        except Exception as e:
            self.logger.error(f"Error reading histogram for {symbol}: {str(e)}")
        return None

    def _get_gauge_value(self, metric, model_type, symbol):
        try:
            for s in metric.collect()[0].samples:
                if s.labels.get("model_type") == model_type and s.labels.get("symbol") == symbol:
                    return float(s.value)
        except Exception as e:
            self.logger.error(f"Error reading gauge for {symbol}: {str(e)}")
        return None

    async def _trigger_retrain(self, symbol: str, model_type: str):
        try:
            self.logger.warning(f"Triggering retrain for {symbol} ({model_type})")
            start = datetime.now() - timedelta(days=730)
            end = datetime.now()
            await self.training_service.train_model(
                symbol=symbol,
                model_type=model_type,
                start_date=start,
                end_date=end,
            )
        except Exception as e:
            self.logger.error(f"Failed retraining {symbol} ({model_type}): {str(e)}")